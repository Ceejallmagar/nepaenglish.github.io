"""Convert the pinned float16 CTranslate2 Whisper Small weights for local training."""

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np
import torch


def tensor_locations(path):
    entries = {}
    with path.open("rb") as stream:
        def read(code):
            return struct.unpack("<" + code, stream.read(struct.calcsize("<" + code)))[0]

        def name():
            return stream.read(read("H")).rstrip(b"\0").decode("utf-8")

        if (read("I"), name(), read("I")) != (6, "WhisperSpec", 3):
            raise ValueError("Expected the demo's pinned CTranslate2 Whisper Small format")
        for _ in range(read("I")):
            key = name()
            shape = tuple(read("I") for _ in range(read("B")))
            dtype = read("B")
            size = read("I")
            if dtype == 4:
                if size != 2 * np.prod(shape):
                    raise ValueError(f"Unexpected float16 weight size: {key}")
                entries[key] = (stream.tell(), shape)
            stream.seek(size, 1)
    return entries


def weight_mappings():
    yield "encoder.positional_embedding", "encoder/position_encodings/encodings", slice(None)
    yield "decoder.positional_embedding", "decoder/position_encodings/encodings", slice(None)
    yield "decoder.token_embedding.weight", "decoder/embeddings/weight", slice(None)
    for conv in ("conv1", "conv2"):
        for field in ("weight", "bias"):
            yield f"encoder.{conv}.{field}", f"encoder/{conv}/{field}", slice(None)
    for side, final_norm in (("encoder", "ln_post"), ("decoder", "ln")):
        for field, ct2_field in (("weight", "gamma"), ("bias", "beta")):
            yield f"{side}.{final_norm}.{field}", f"{side}/layer_norm/{ct2_field}", slice(None)
        for layer in range(12):
            native = f"{side}.blocks.{layer}"
            ct2 = f"{side}/layer_{layer}"
            for field, ct2_field in (("weight", "gamma"), ("bias", "beta")):
                yield f"{native}.mlp_ln.{field}", f"{ct2}/ffn/layer_norm/{ct2_field}", slice(None)
                yield f"{native}.attn_ln.{field}", f"{ct2}/self_attention/layer_norm/{ct2_field}", slice(None)
                if side == "decoder":
                    yield f"{native}.cross_attn_ln.{field}", f"{ct2}/attention/layer_norm/{ct2_field}", slice(None)
            for field in ("weight", "bias"):
                for index in (0, 2):
                    yield f"{native}.mlp.{index}.{field}", f"{ct2}/ffn/linear_{index // 2}/{field}", slice(None)
                for index, name in enumerate(("query", "key", "value")):
                    if name != "key" or field == "weight":
                        rows = slice(index * 768, (index + 1) * 768)
                        yield f"{native}.attn.{name}.{field}", f"{ct2}/self_attention/linear_0/{field}", rows
                        if side == "decoder":
                            cross_index = 0 if name == "query" else 1
                            cross_rows = slice(None) if name == "query" else slice((index - 1) * 768, index * 768)
                            yield f"{native}.cross_attn.{name}.{field}", f"{ct2}/attention/linear_{cross_index}/{field}", cross_rows
                yield f"{native}.attn.out.{field}", f"{ct2}/self_attention/linear_1/{field}", slice(None)
                if side == "decoder":
                    yield f"{native}.cross_attn.out.{field}", f"{ct2}/attention/linear_2/{field}", slice(None)


def as_native(source, output):
    locations = tensor_locations(source / "model.bin")
    dims = dict(n_mels=80, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12,
                n_audio_layer=12, n_vocab=51865, n_text_ctx=448, n_text_state=768,
                n_text_head=12, n_text_layer=12)
    weights = {}
    for native, ct2, rows in weight_mappings():
        offset, shape = locations[ct2]
        array = np.memmap(source / "model.bin", dtype="<f2", mode="r", offset=offset, shape=shape)
        weights[native] = torch.from_numpy(array[rows].copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".tmp")
    torch.save({"dims": dims, "model_state_dict": weights}, pending)
    pending.replace(output)


def as_ct2(source, output, checkpoint):
    weights = torch.load(checkpoint, mmap=True, weights_only=True)["model_state_dict"]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "vocabulary.txt"):
        shutil.copyfile(source / name, output / name)
    model = output / "model.bin"
    pending = model.with_suffix(".tmp")
    shutil.copyfile(source / "model.bin", pending)
    locations = tensor_locations(pending)
    for native, ct2, rows in weight_mappings():
        if native.startswith("decoder.") and native.endswith(("query.weight", "value.weight")):
            offset, shape = locations[ct2]
            array = np.memmap(pending, dtype="<f2", mode="r+", offset=offset, shape=shape)
            array[rows] = weights[native].numpy()
            del array
    pending.replace(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--export", type=Path, metavar="MERGED_CHECKPOINT")
    args = parser.parse_args()
    if args.export:
        as_ct2(args.source, args.output, args.export)
    else:
        as_native(args.source, args.output)


if __name__ == "__main__":
    main()
