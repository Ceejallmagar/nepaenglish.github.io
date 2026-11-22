"""Run Shruti Nepali through the existing local caption server."""

import os

from cactus import Whistle
from whisperlivekit.timed_objects import ASRToken

from word_timestamp_backend import WordTimestampASR, serve_word_timestamp_asr


class ShrutiASR(WordTimestampASR):
    backend_choice = "shruti"

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        return Whistle(model_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.model.transcribe(audio, language=self.original_language, word_timestamps=True)
        return [
            ASRToken(word["start"], word["end"], " " + word["word"].strip(),
                     detected_language=result.get("language"), probability=word.get("probability"))
            for word in result.get("words", []) if word["word"].strip()
        ]


def serve_shruti():
    serve_word_timestamp_asr(
        ShrutiASR,
        model_size="shruti-nepali",
        model_path=os.environ["NEPALISH_SHRUTI_WEIGHTS"],
    )
