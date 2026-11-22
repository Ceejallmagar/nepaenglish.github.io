# Whisper Small fine-tuning

The [Nepali–English technical-interview dataset](https://huggingface.co/datasets/devrahulbanjara/ne-en-codeswitching-asr-technical-interview) contains people reading AI-written scripts. The published splits contain 639 training clips, 79 validation clips, and 81 test clips. The card lists Apache 2.0. Use the result as a technical-speech experiment and measure everyday conversation on a separate recording set.

## Prepare and run

Run the demo setup first. On a fresh machine, use the Hugging Face `hf` command to download the pinned split files:

```bash
hf download devrahulbanjara/ne-en-codeswitching-asr-technical-interview \
  --repo-type dataset --revision 06feb6aef040848bf25febc9a354612feab4e8e0 \
  --include 'data/*.parquet' --local-dir training-data/technical-interviews/download

for split in train validation test; do
  cp "training-data/technical-interviews/download/data/$split-00000-of-00001.parquet" \
    "training-data/technical-interviews/$split.parquet"
done

python3 training/prepare.py training-data/technical-interviews
.venv/bin/python training/ct2_checkpoint.py models/whisper-small models/training-source/small-from-ct2.pt
```

The Python used for `prepare.py` needs `pyarrow`. The trainer uses the demo's virtual environment. Its trainable base is reconstructed from the pinned float16 Whisper Small model already installed by the demo. It freezes the encoder and original decoder weights and trains rank-8 LoRA adapters on the decoder attention query and value projections.

```bash
DATA=training-data/technical-interviews
BASE=models/training-source/small-from-ct2.pt
OUT=training-runs/technical-interviews

.venv/bin/python training/train.py train --model "$BASE" --data "$DATA" --output "$OUT"
.venv/bin/python training/train.py baseline --model "$BASE" --data "$DATA" --output "$OUT"
.venv/bin/python training/train.py tuned --model "$OUT/whisper-small-ne-en.pt" --data "$DATA" --output "$OUT"

.venv/bin/python training/ct2_checkpoint.py models/whisper-small "$OUT/ct2" \
  --export "$OUT/whisper-small-ne-en.pt"

OMP_NUM_THREADS=4 .venv/bin/python -m training.score_ct2 --model models/whisper-small \
  --data "$DATA" --output "$OUT/baseline-live-decoder-test.json"
OMP_NUM_THREADS=4 .venv/bin/python -m training.score_ct2 --model "$OUT/ct2" \
  --data "$DATA" --output "$OUT/tuned-live-decoder-test.json"

NEPALISH_MODEL_DIR="$OUT/ct2" NEPALISH_PORT=8766 ./run.sh
```

Training saves a resumable state every 20 clips by default. Repeat the same training command to resume. A finished run writes `adapter.pt`, a merged `whisper-small-ne-en.pt`, and JSON evaluation reports for the stock and tuned models. The `score_ct2` reports run the app's decoder on complete test clips. They include word error rate and error rate after filtering to words containing Latin letters. The latter is a proxy for English word and script errors.
