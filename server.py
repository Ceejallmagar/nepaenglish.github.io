import os

from whisperlivekit.basic_server import main
from whisperlivekit.local_agreement import whisper_online

from whisper_runtime import load_cpu_whisper, transcribe_audio


class CpuWhisperASR(whisper_online.FasterWhisperASR):
    reset_between_segments = False

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        return load_cpu_whisper(model_dir)

    def transcribe(self, audio, init_prompt=""):
        return transcribe_audio(
            self.model,
            audio,
            language=self.original_language,
            prompt=init_prompt.strip() or None,
            reset_between_segments=self.reset_between_segments,
            **self.transcribe_kargs,
        )


class NepTransASR(CpuWhisperASR):
    # Fresh decoder context limits repetition on pauses and long recordings.
    reset_between_segments = True


if __name__ == "__main__":
    # Keep the bundled browser UI and streaming policy, with a CPU-only decoder.
    whisper_online.FasterWhisperASR = (
        NepTransASR if os.environ.get("NEPALISH_ENGINE") == "neptrans" else CpuWhisperASR
    )
    main()
