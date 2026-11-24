# Nepali–English live captions

An offline microphone transcriber for Linux. Speak Nepali and English in the same sentence and see captions in your browser. The recommended engine, IndicConformer, writes Nepali in Devanagari and English in Latin script. WhisperLiveKit streams the audio to a server on your own machine.

## Quick start

Install `uv` and FFmpeg, then run these commands from the repository root. Setup needs an internet connection to install the Python 3.12 environments and download the models. Subsequent launches use local weights.

```bash
./setup.sh --indicconformer
./run.sh
```

Open the caption page, click the microphone, and allow microphone access. Try saying “आजको meeting मा मैले new feature demo गरेँ।” Click the microphone again to finish, then select and copy the captions. Press `Ctrl+C` in the terminal to stop the server. [Open local captions](http://127.0.0.1:8765)

## Measured word error

Across **37 public podcast clips**, stock IndicConformer made **215 word edits against 732 reference words**, for a **29.37% word error rate (WER)**. On the 24 clips containing both languages, its WER was **26.56%**. The table uses complete audio clips. The live microphone path is measured separately below. [Podcast dataset](https://huggingface.co/datasets/SamaFiroz/nepali-english-codeswitch-podcasts)

| CPU decoder | Word edits / 732 | Overall WER | Nepali only · 13 clips | Mixed · 24 clips |
| --- | ---: | ---: | ---: | ---: |
| IndicConformer, default | 215 | 29.37% | 35.91% | 26.56% |
| IndicConformer, technical vocabulary | 217 | 29.64% | 35.91% | 26.95% |
| NepTrans v4 | 354 | 48.36% | 57.73% | 44.34% |
| Shruti Nepali | 369 | 50.41% | 50.00% | 50.59% |

WER counts the insertions, deletions and substitutions needed to match the reference, divided by the total number of reference words. The scorer ignores case and punctuation. A spacing or script difference can still count as a word error. [Scoring code](training/asr_metrics.py)

In a separate live replay, the technical vocabulary corrected `blueth head` to `bluetooth headset`. On the podcast set it added **two edits overall** (+0.27 percentage points). A width-four beam search also increased WER to **29.78%**. The individual speaker recording has a machine-assisted draft transcript, so its exact WER is unknown.

## Faster live captions

Start IndicConformer with a one-second audio buffer to request partial captions sooner. The default waits for two seconds of accumulated audio. Shorter intervals run the decoder more often.

```bash
NEPALISH_ENGINE=indicconformer NEPALISH_LOW_LATENCY=1 ./run.sh
```

On a Ryzen 7 8840HS using four CPU threads, the first caption appeared about **one second sooner** with the shorter buffer. Both runs replayed the same six labeled clips through the microphone WebSocket with stock IndicConformer. Model startup is excluded. First caption time includes partial text. Finalization time starts when the audio ends.

| Audio accumulation | Median first caption | Median finalization | Final word edits / reference words | Live WER |
| --- | ---: | ---: | ---: | ---: |
| Default · 2 s | 2.23 s | 0.80 s | 52 / 167 | 31.14% |
| Low latency · 1 s | 1.22 s | 0.73 s | 51 / 167 | 30.54% |

The one-edit difference is small. These six clips measure the latency tradeoff on this machine. Longer conversations and other CPUs may give different results.

## Technical vocabulary

The bundled vocabulary contains eight entries, including `bluetooth headset` and `microphone`. To use it, run the command below. For your own terms, point the variable at a UTF-8 file with one word or phrase per line and restart the server. [Example vocabulary](examples/technical-terms.txt)

```bash
NEPALISH_ENGINE=indicconformer NEPALISH_KEYWORDS_FILE=examples/technical-terms.txt ./run.sh
```

## Other launch options

| Use case | Command |
| --- | --- |
| All-English speech with Whisper Small | `NEPALISH_ENGINE=whisper NEPALISH_LANGUAGE=en ./run.sh` |
| Shruti Nepali | `NEPALISH_ENGINE=shruti ./run.sh` |
| Another local port | `NEPALISH_PORT=8766 ./run.sh` |
| Custom Whisper model directory | `NEPALISH_ENGINE=whisper NEPALISH_MODEL_DIR=/path/to/model ./run.sh` |

Base setup (`./setup.sh`) installs Shruti and multilingual Whisper Small. The `--indicconformer` option adds a separate runtime and about 514 MB of IndicConformer weights. The checkpoint is licensed under **CC BY-NC 4.0** for noncommercial use. [IndicConformer model card](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) · [Shruti model card](https://huggingface.co/udgamlabs/shruti-nepali)

For NepTrans v4, run `./setup.sh --neptrans` followed by `NEPALISH_ENGINE=neptrans ./run.sh`. Its weights add about 1.55 GB. It took 489 seconds to decode 228 seconds of podcast audio on the benchmark laptop. [NepTrans v4 model card](https://huggingface.co/DontHash/neptrans-v4-ct2)

## Reproduce the benchmarks

Use a local manifest and 16 kHz mono audio. The manifest is a JSON array with an `id`, relative `audio` path, reference `text`, `group` (`nepali` or `mixed`), and `duration` in seconds for each clip. Benchmark audio and local reports are excluded from Git.

```bash
MANIFEST=training-data/conversation-eval/manifest.json

.venv-indicconformer/bin/python -m training.score_conversation indicconformer \
  --model models/indicconformer-ne-en --manifest "$MANIFEST" \
  --output training-runs/podcast-default.json
```

For the vocabulary comparison, add `--keywords-file examples/technical-terms.txt` and choose a different output file. To measure live latency, start the server in the desired mode and run this in a second terminal. The live scorer replays three Nepali-only and three mixed clips through the microphone WebSocket.

```bash
.venv/bin/python -m training.score_live --port 8765 \
  --manifest training-data/conversation-eval/manifest.json \
  --output training-runs/live.json
```

The repository also includes a separate Whisper Small fine-tuning experiment for technical interviews. [Training guide](training/README.md)
