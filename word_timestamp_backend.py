"""Serve ASR models that already return individual timestamped words."""

import os

import uvicorn
from whisperlivekit import core
from whisperlivekit.basic_server import create_app
from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.local_agreement.backends import ASRBase


class WordTimestampASR(ASRBase):
    sep = ""

    def ts_words(self, tokens):
        return tokens

    def segments_end_ts(self, tokens):
        return [tokens[-1].end] if tokens else []


def serve_word_timestamp_asr(backend, *, model_size, model_path):
    def build_backend(*, lan, model_dir, buffer_trimming, buffer_trimming_sec,
                      confidence_validation, **kwargs):
        asr = backend(lan=lan, model_dir=model_dir)
        asr.buffer_trimming = buffer_trimming
        asr.buffer_trimming_sec = buffer_trimming_sec
        asr.confidence_validation = confidence_validation
        asr.tokenizer = None
        return asr

    core.backend_factory = build_backend
    config = WhisperLiveKitConfig(
        host="127.0.0.1",
        port=int(os.environ["NEPALISH_PORT"]),
        model_size=model_size,
        model_dir=model_path,
        backend=backend.backend_choice,
        backend_policy="localagreement",
        lan=os.environ["NEPALISH_LANGUAGE"],
        min_chunk_size=0.5,
        asr_coalesce_min_s=2,
        log_level="WARNING",
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port)
