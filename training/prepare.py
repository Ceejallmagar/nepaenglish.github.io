"""Extract the official technical-interview splits without changing their transcripts."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()

    for split, expected_rows in (("train", 639), ("validation", 79), ("test", 81)):
        if args.split and split != args.split:
            continue
        source = parquet.ParquetFile(args.data_dir / f"{split}.parquet")
        if source.metadata.num_rows != expected_rows:
            raise ValueError(f"Unexpected {split} row count: {source.metadata.num_rows}")
        audio_dir = args.data_dir / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest = args.data_dir / f"{split}.jsonl"
        pending = manifest.with_suffix(".tmp")
        seen = set()
        with pending.open("w", encoding="utf-8") as stream:
            for batch in source.iter_batches(batch_size=16, columns=["id", "audio", "transcript"]):
                for row in batch.to_pylist():
                    clip_id = int(row["id"])
                    if clip_id in seen or not row["transcript"].strip() or not row["audio"]["bytes"]:
                        raise ValueError(f"Invalid or duplicate {split} clip {clip_id}")
                    seen.add(clip_id)
                    path = audio_dir / f"{clip_id}.wav"
                    path.write_bytes(row["audio"]["bytes"])
                    clip = {"id": clip_id, "audio": str(path.relative_to(args.data_dir)), "text": row["transcript"]}
                    stream.write(json.dumps(clip, ensure_ascii=False) + "\n")
        pending.replace(manifest)
        print(f"{split}: extracted {len(seen)} clips", flush=True)


if __name__ == "__main__":
    main()
