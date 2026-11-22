# Nepali–English live captions

A local Linux demo for Nepali and English speech in the same conversation. WhisperLiveKit captures microphone audio and streams captions. The demo uses IndicConformer for Nepali-led conversations when installed and uses Shruti Nepali otherwise.

## Run the demo

If setup has already run in this workspace, start here:

```bash
./run.sh
```

Open [the local caption page](http://127.0.0.1:8765), click the microphone button, and allow microphone access. Try saying “आजको meeting मा मैले new feature demo गरेँ।” Click the microphone again to finish. You can select and copy the finished captions into a text file. Press `Ctrl+C` in the terminal to shut down the server.

The demo starts in Nepali mode for Nepali-led mixed speech. For an all-English recording, run `NEPALISH_LANGUAGE=en ./run.sh`. To use another local port, run `NEPALISH_PORT=8766 ./run.sh`.

## Set up a fresh checkout

With `uv` and FFmpeg installed, run `./setup.sh --indicconformer` once while online. It installs the IndicConformer runtime in its own Python 3.12 environment and downloads about 514 MB of model weights. The setup also installs the [Shruti Nepali](https://huggingface.co/udgamlabs/shruti-nepali) model, its Cactus runtime, and multilingual Whisper Small. Run `./setup.sh` for just these base dependencies and models. After setup, `./run.sh` uses local weights and binds to the loopback interface.

Run `NEPALISH_ENGINE=shruti ./run.sh` for Shruti or `NEPALISH_ENGINE=whisper ./run.sh` for stock Whisper Small. For a custom model directory, set `NEPALISH_MODEL_DIR` and choose its engine explicitly.

## Model comparison

[AI Lab's IndicConformer](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) was trained for Nepali–English code-switching. It writes Nepali in Devanagari and English in lowercase Latin script, with minimal punctuation. The checkpoint is licensed under CC BY-NC 4.0 for noncommercial use.

On the same 37 public [podcast clips](https://huggingface.co/datasets/SamaFiroz/nepali-english-codeswitch-podcasts), lower word error rate is better.

| CPU model | All clips | Nepali-only | Mixed Nepali–English |
| --- | ---: | ---: | ---: |
| Shruti Nepali | 50.4% | 50.0% | 50.6% |
| NepTrans v4 | 48.4% | 57.7% | 44.3% |
| IndicConformer | 29.4% | 35.9% | 26.6% |

[NepTrans v4](https://huggingface.co/DontHash/neptrans-v4-ct2) is a larger offline model trained on conversational and read Nepali, including Nepali–English speech. It runs in CPU INT8 mode. The download is about 1.55 GB.

```bash
./setup.sh --neptrans
NEPALISH_ENGINE=neptrans NEPALISH_PORT=8766 ./run.sh
```

NepTrans took longer than the clips themselves to decode on this laptop. To repeat the IndicConformer benchmark if the clips are available in `training-data/conversation-eval`, run the following command.

```bash
.venv-indicconformer/bin/python -m training.score_conversation indicconformer \
  --model models/indicconformer-ne-en \
  --manifest training-data/conversation-eval/manifest.json \
  --output training-runs/conversation-models/indicconformer.json
```

See [the training guide](training/README.md) for the Nepali–English technical-interview fine-tuning run and how to launch a trained model on a separate port.
