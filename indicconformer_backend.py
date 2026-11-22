"""Run the Nepali–English IndicConformer on local CPU audio."""

import os
from threading import Lock

import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils import logging as nemo_logging
from whisperlivekit.timed_objects import ASRToken

from word_timestamp_backend import WordTimestampASR, serve_word_timestamp_asr


def load_indicconformer(model_path):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    return ASRModel.restore_from(str(model_path), map_location=torch.device("cpu")).eval()


def transcribe_indicconformer(model, audio):
    return model.transcribe([audio], batch_size=1, num_workers=0, timestamps=True, verbose=False)[0]


class IndicConformerASR(WordTimestampASR):
    backend_choice = "indicconformer"

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        self.decode_lock = Lock()
        return load_indicconformer(model_dir)

    def transcribe(self, audio, init_prompt=""):
        if not len(audio):
            return []
        with self.decode_lock:
            hypothesis = transcribe_indicconformer(self.model, audio)
        return [
            ASRToken(word["start"], word["end"], " " + word["word"].strip())
            for word in hypothesis.timestamp.get("word", [])
            if word["word"].strip()
        ]


if __name__ == "__main__":
    nemo_logging.setLevel(nemo_logging.ERROR)
    serve_word_timestamp_asr(
        IndicConformerASR,
        model_size="indicconformer-ne-en",
        model_path=os.environ["NEPALISH_CONFORMER_WEIGHTS"],
    )
