"""Compare offline caption engines on the same unmodified audio clips."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from training.asr_metrics import score


def create_decoder(engine, directory, language, *, short_windows=False):
    if engine == "shruti":
        from cactus import Whistle

        model = Whistle(directory / "shruti-nepali.cact")

        def decode(audio):
            return model.transcribe(audio, language=language, word_timestamps=True)["text"]
    elif engine == "indicconformer":
        from indicconformer_backend import load_indicconformer, transcribe_indicconformer

        model = load_indicconformer(directory / "indicconformer-ne-en-codemix.nemo")

        def decode(audio):
            return transcribe_indicconformer(model, audio).text
    else:
        from whisper_runtime import load_cpu_whisper, transcribe_audio

        model = load_cpu_whisper(directory, short_windows=short_windows)

        def decode(audio):
            segments = transcribe_audio(model, audio, language=language, reset_between_segments=True)
            return "".join(segment.text for segment in segments).strip()

    return decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("shruti", "neptrans", "indicconformer"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ne")
    parser.add_argument("--short-windows", action="store_true", help="Try shorter NepTrans encoder windows")
    args = parser.parse_args()
    if args.short_windows and args.engine != "neptrans":
        parser.error("--short-windows applies to NepTrans")
    decode = create_decoder(args.engine, args.model, args.language, short_windows=args.short_windows)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    seconds = 0.0
    for index, row in enumerate(rows, start=1):
        path = args.manifest.parent / row["audio"]
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected 16 kHz mono audio in {path}")
        start = time.monotonic()
        row["prediction"] = decode(audio)
        row["decode_seconds"] = time.monotonic() - start
        seconds += len(audio) / rate
        if index % 5 == 0 or index == len(rows):
            print(f"{args.engine}: {index}/{len(rows)} clips", flush=True)

    groups = {"all": rows}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    scores = {
        label: {"clips": len(items), **score([row["text"] for row in items], [row["prediction"] for row in items])}
        for label, items in groups.items()
    }
    elapsed = sum(row["decode_seconds"] for row in rows)
    report = {"model": args.engine, "language": args.language, "short_windows": args.short_windows, "audio_seconds": seconds,
              "decode_seconds": elapsed, "real_time_factor": elapsed / seconds, "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**report, "predictions": rows}, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
