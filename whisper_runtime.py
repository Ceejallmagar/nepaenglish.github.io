"""CPU decoder shared by the live app and its audio benchmark."""

import numpy as np
from faster_whisper import WhisperModel


class ShortWindowWhisper(WhisperModel):
    def encode(self, features):
        # Faster-whisper pads every chunk to 30 seconds. Keep at least a
        # 12-second encoder window and two seconds of padding for short speech.
        frame_axes = tuple(range(features.ndim - 1))
        occupied = np.flatnonzero(np.any(features != 0, axis=frame_axes))
        content = int(occupied[-1]) + 1 if occupied.size else 0
        frames = max(1200, ((content + 399) // 200) * 200)
        return super().encode(np.ascontiguousarray(features[..., :frames]))


def load_cpu_whisper(model_dir, *, short_windows=False):
    decoder = ShortWindowWhisper if short_windows else WhisperModel
    return decoder(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)


def transcribe_audio(model, audio, *, language, prompt=None, reset_between_segments=False, **options):
    segments, _ = model.transcribe(
        audio,
        language=language,
        # An empty string still adds a prompt token in faster-whisper.
        initial_prompt=prompt,
        word_timestamps=True,
        beam_size=1,
        temperature=0.0,
        repetition_penalty=1.12,
        no_repeat_ngram_size=5,
        max_new_tokens=192,
        condition_on_previous_text=not reset_between_segments,
        **options,
    )
    return list(segments)
