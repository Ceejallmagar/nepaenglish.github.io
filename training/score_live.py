"""Replay benchmark clips through the same WebSocket used by the microphone."""

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from statistics import median

from websockets.asyncio.client import connect

from training.asr_metrics import score


async def replay_clip(url, row, audio_path):
    audio = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-c:a", "libopus", "-f", "webm", "pipe:1",
    ])
    async with connect(url, open_timeout=10) as socket:
        config = json.loads(await socket.recv())
        if config.get("type") != "config" or config.get("useAudioWorklet"):
            raise RuntimeError(f"Expected a WebM audio stream, got {config}")
        started = time.monotonic()

        async def send_audio():
            chunk_size = 1600
            chunks = range(0, len(audio), chunk_size)
            for offset in chunks:
                await socket.send(audio[offset:offset + chunk_size])
                await asyncio.sleep(row["duration"] / len(chunks))
            ended = time.monotonic()
            await socket.send(b"")
            return ended

        sending = asyncio.create_task(send_audio())
        first_caption = None
        transcript = ""
        try:
            async with asyncio.timeout(90):
                async for message in socket:
                    response = json.loads(message)
                    if response.get("type") == "error":
                        raise RuntimeError(response)
                    if response.get("type") == "ready_to_stop":
                        break
                    current = " ".join(
                        line["text"] for line in response.get("lines", []) if line.get("text")
                    ).strip()
                    if first_caption is None and (current or response.get("buffer_transcription", "").strip()):
                        first_caption = time.monotonic() - started
                    if current:
                        transcript = current
                else:
                    raise RuntimeError("Server disconnected before finalizing the transcript")
                ended = await sending
        finally:
            if not sending.done():
                sending.cancel()

        return {
            **row,
            "prediction": transcript,
            "first_caption_seconds": first_caption,
            "final_after_end_seconds": time.monotonic() - ended,
        }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = []
    for group in ("nepali", "mixed"):
        eligible = [row for row in rows if row["group"] == group and 5 <= row["duration"] < 12]
        if len(eligible) < 3:
            parser.error(f"At least three clips from five to twelve seconds are required for {group}")
        clips.extend(eligible[:3])

    predictions = []
    for row in clips:
        result = await replay_clip(
            f"ws://127.0.0.1:{args.port}/asr", row, args.manifest.parent / row["audio"]
        )
        predictions.append(result)
        print(f"{row['id']}: {result['prediction']}", flush=True)

    first_captions = [row["first_caption_seconds"] for row in predictions if row["first_caption_seconds"] is not None]
    report = {
        "clips": len(clips),
        **score([row["text"] for row in predictions], [row["prediction"] for row in predictions]),
        "clips_with_captions": len(first_captions),
        "median_first_caption_seconds": median(first_captions) if first_captions else None,
        "median_final_after_end_seconds": median(row["final_after_end_seconds"] for row in predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**report, "predictions": predictions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
