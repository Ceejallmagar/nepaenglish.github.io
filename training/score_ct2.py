"""Score a Whisper checkpoint with the same decoder settings as the live demo."""

import argparse
import json
import re
from pathlib import Path

from server import CpuWhisperASR
from training.train import read_audio, read_clips, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    clips = read_clips(args.data, args.split)
    asr = CpuWhisperASR(lan="ne", model_dir=str(args.model))
    rows = []
    for index, clip in enumerate(clips, start=1):
        segments = asr.transcribe(read_audio(args.data / clip["audio"]))
        rows.append({"id": clip["id"], "reference": clip["text"], "prediction": "".join(seg.text for seg in segments).strip()})
        if index % 10 == 0 or index == len(clips):
            print(f"{args.split}: {index}/{len(clips)} clips", flush=True)

    def scores_for(items):
        return {"clips": len(items), **score([row["reference"] for row in items], [row["prediction"] for row in items])}

    mixed = [row for row in rows if re.search(r"[\u0900-\u097f]", row["reference"]) and re.search(r"[a-zA-Z]", row["reference"])]
    report = {"split": args.split, **scores_for(rows), "mixed_script_references": scores_for(mixed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(".tmp")
    pending.write_text(json.dumps({**report, "predictions": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
