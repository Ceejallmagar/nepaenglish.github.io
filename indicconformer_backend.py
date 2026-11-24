"""Run the Nepali–English IndicConformer on local CPU audio."""

import os
from pathlib import Path
from threading import Lock

import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils import logging as nemo_logging
from omegaconf import OmegaConf
from whisperlivekit.timed_objects import ASRToken

from word_timestamp_backend import WordTimestampASR, serve_word_timestamp_asr


def load_indicconformer(model_path, *, keywords_file=None):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    model = ASRModel.restore_from(str(model_path), map_location=torch.device("cpu")).eval()
    if keywords_file:
        phrases = [line.strip().lower() for line in Path(keywords_file).read_text(encoding="utf-8").splitlines()]
        phrases = list(dict.fromkeys(phrase for phrase in phrases if phrase))
        if phrases:
            config = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding))
            config.greedy.boosting_tree.key_phrases_list = phrases
            config.greedy.boosting_tree.use_triton = False
            config.greedy.boosting_tree_alpha = 4.0
            model.change_decoding_strategy(config, decoder_type="rnnt", verbose=False)
    return model


def transcribe_indicconformer(model, audio):
    return model.transcribe([audio], batch_size=1, num_workers=0, timestamps=True, verbose=False)[0]


class IndicConformerASR(WordTimestampASR):
    backend_choice = "indicconformer"

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        self.decode_lock = Lock()
        return load_indicconformer(model_dir, keywords_file=os.environ.get("NEPALISH_KEYWORDS_FILE"))

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
