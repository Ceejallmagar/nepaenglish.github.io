"""Fine-tune multilingual Whisper Small on CPU with decoder LoRA."""

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from whisperlivekit.whisper.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
from whisperlivekit.whisper.decoding import DecodingOptions, decode
from whisperlivekit.whisper.model import ModelDimensions, MultiHeadAttention, Whisper
from whisperlivekit.whisper.tokenizer import get_tokenizer

if __package__:
    from .asr_metrics import score
else:
    from asr_metrics import score


class LoRAProjection(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, audio):
        return self.base(audio) + self.up(self.down(audio)) * self.scale


def add_adapters(model, rank, alpha):
    for attention in list(model.decoder.modules()):
        if isinstance(attention, MultiHeadAttention):
            attention.query = LoRAProjection(attention.query, rank, alpha)
            attention.value = LoRAProjection(attention.value, rank, alpha)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims).requires_grad_(False).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def read_clips(directory, split):
    with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        factor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // factor, sample_rate // factor)
    if not 0 < len(audio) <= N_SAMPLES:
        raise ValueError(f"Expected a single nonempty Whisper window: {path} ({len(audio) / SAMPLE_RATE:.2f}s)")
    return audio


def read_mel(path):
    return log_mel_spectrogram(pad_or_trim(read_audio(path)))


def save_progress(path, model, optimizer, next_step, mean_loss, settings):
    pending = path.with_suffix(".tmp")
    torch.save({
        "adapters": {name: param.detach() for name, param in model.named_parameters() if param.requires_grad},
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "mean_loss": mean_loss,
        "settings": settings,
    }, pending)
    pending.replace(path)


def train(args, model, tokenizer, clips, steps):
    settings = {"seed": args.seed, "rank": args.rank, "alpha": args.alpha, "learning_rate": args.learning_rate}
    torch.manual_seed(args.seed)
    add_adapters(model, args.rank, args.alpha)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=args.learning_rate, weight_decay=0.0)
    progress_path = args.output / "training-state.pt"
    step, mean_loss = 0, 0.0
    if progress_path.exists():
        state = torch.load(progress_path, map_location="cpu", weights_only=True)
        if state["settings"] != settings or params.keys() != state["adapters"].keys():
            raise ValueError("Training settings changed. Use a new output directory.")
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(state["adapters"][name])
        optimizer.load_state_dict(state["optimizer"])
        step, mean_loss = state["next_step"], state["mean_loss"]

    order = list(clips)
    random.Random(args.seed).shuffle(order)
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    started = time.monotonic()
    print(f"Trainable parameters: {sum(p.numel() for p in params.values()):,}; steps: {step} → {steps}", flush=True)
    for index in range(step, steps):
        clip = order[index]
        text = unicodedata.normalize("NFC", clip["text"].strip())
        tokens = prefix + tokenizer.encode(" " + text) + [tokenizer.eot]
        if len(tokens) > model.dims.n_text_ctx:
            raise ValueError(f"Clip {clip['id']} has too many text tokens")
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            features = model.encoder(read_mel(args.data / clip["audio"]).unsqueeze(0))
        optimizer.zero_grad(set_to_none=True)
        logits = model.decoder(tokens[:, :-1], features)
        offset = len(prefix) - 1
        loss = F.cross_entropy(logits[:, offset:, :].flatten(0, 1), tokens[:, offset + 1:].flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss on clip {clip['id']}")
        loss.backward()
        nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
        optimizer.step()
        completed = index + 1
        mean_loss += (loss.item() - mean_loss) / completed
        if completed % 10 == 0 or completed == steps:
            elapsed = time.monotonic() - started
            print(f"Train {completed}/{steps} | mean loss {mean_loss:.3f} | {elapsed / (completed - step):.2f}s/clip", flush=True)
        if completed % args.save_every == 0 or completed == steps:
            save_progress(progress_path, model, optimizer, completed, mean_loss, settings)

    torch.save({"base": "openai/whisper-small", "settings": settings, "steps": max(step, steps),
                "adapters": {name: param.detach() for name, param in params.items()}}, args.output / "adapter.pt")


def merge_adapters(model):
    with torch.no_grad():
        for attention in model.decoder.modules():
            if isinstance(attention, MultiHeadAttention):
                for name in ("query", "value"):
                    adapter = getattr(attention, name)
                    adapter.base.weight.add_(adapter.up.weight @ adapter.down.weight, alpha=adapter.scale)
                    setattr(attention, name, adapter.base)


@torch.inference_mode()
def evaluate(args, model, split):
    clips = read_clips(args.data, split)
    if args.eval_limit:
        clips = random.Random(args.seed).sample(clips, min(args.eval_limit, len(clips)))
    options = DecodingOptions(language="ne", fp16=False, without_timestamps=True, sample_len=220)
    predictions = []
    for index in range(0, len(clips), args.eval_batch_size):
        batch = clips[index:index + args.eval_batch_size]
        mels = torch.stack([read_mel(args.data / clip["audio"]) for clip in batch])
        predictions.extend(result.text for result in decode(model, mels, options))
        if len(predictions) % 10 == 0 or len(predictions) == len(clips):
            print(f"{args.mode} {split}: {len(predictions)}/{len(clips)} clips", flush=True)
    references = [clip["text"] for clip in clips]
    report = {"split": split, "clips": len(clips), **score(references, predictions),
              "predictions": [{"id": clip["id"], "reference": ref, "prediction": pred} for clip, ref, pred in zip(clips, references, predictions, strict=True)]}
    path = args.output / f"{args.mode}-{split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "baseline", "tuned"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, help="Limit training to a prefix of one shuffled epoch")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    MultiHeadAttention.use_sdpa = True
    model = load_model(args.model)

    if args.mode == "train":
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="ne", task="transcribe")
        clips = read_clips(args.data, "train")
        steps = args.steps if args.steps is not None else len(clips)
        if not 0 < steps <= len(clips):
            raise ValueError(f"Steps must be between 1 and {len(clips)}")
        train(args, model, tokenizer, clips, steps)
        merge_adapters(model)
        pending = args.output / "whisper-small-ne-en.pt.tmp"
        torch.save({"dims": vars(model.dims), "model_state_dict": {key: value.half() for key, value in model.state_dict().items()}}, pending)
        pending.replace(args.output / "whisper-small-ne-en.pt")
    else:
        evaluate(args, model, args.split)


if __name__ == "__main__":
    main()
