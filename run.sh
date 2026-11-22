#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if ! command -v ffmpeg >/dev/null; then
    echo 'Install FFmpeg to decode microphone audio.' >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    echo 'Run ./setup.sh before starting the demo.' >&2
    exit 1
fi

shruti_dir="$PWD/models/shruti-nepali"
conformer_dir="$PWD/models/indicconformer-ne-en"
conformer_python=.venv-indicconformer/bin/python
conformer_filename=indicconformer-ne-en-codemix.nemo
engine="${NEPALISH_ENGINE:-whisper}"
export NEPALISH_LANGUAGE="${NEPALISH_LANGUAGE:-ne}"
export NEPALISH_PORT="${NEPALISH_PORT:-8765}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CACTUS_TELEMETRY=0 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

conformer_installed() {
    [[ -x "$conformer_python" ]] && "$conformer_python" -c \
        'import importlib.util; raise SystemExit(importlib.util.find_spec("nemo") is None)'
}

if [[ -z "${NEPALISH_ENGINE:-}" && -z "${NEPALISH_MODEL_DIR:-}" ]]; then
    if [[ "$NEPALISH_LANGUAGE" == ne && -s "$conformer_dir/$conformer_filename" ]] && conformer_installed; then
        engine=indicconformer
    elif [[ -s "$shruti_dir/shruti-nepali.cact" ]]; then
        engine=shruti
    fi
fi

if [[ "$engine" == indicconformer ]]; then
    export NEPALISH_CONFORMER_WEIGHTS="${NEPALISH_MODEL_DIR:-$conformer_dir}/$conformer_filename"
    export WANDB_MODE=disabled
    if [[ ! -s "$NEPALISH_CONFORMER_WEIGHTS" ]] || ! conformer_installed; then
        echo 'Run ./setup.sh --indicconformer to install IndicConformer.' >&2
        exit 1
    fi
    exec "$conformer_python" indicconformer_backend.py
fi

if [[ "$engine" == shruti ]]; then
    export NEPALISH_SHRUTI_WEIGHTS="${NEPALISH_MODEL_DIR:-$shruti_dir}/shruti-nepali.cact"
    if [[ ! -s "$NEPALISH_SHRUTI_WEIGHTS" ]]; then
        echo 'Run ./setup.sh to download the Shruti Nepali weights.' >&2
        exit 1
    fi
    exec .venv/bin/python -c 'from shruti_backend import serve_shruti; serve_shruti()'
fi
case "$engine" in
    whisper)
        model_dir="${NEPALISH_MODEL_DIR:-$PWD/models/whisper-small}"
        model_name=small
        coalesce_seconds=2
        ;;
    neptrans)
        model_dir="${NEPALISH_MODEL_DIR:-$PWD/models/neptrans-v4}"
        model_name=large-v3
        coalesce_seconds=4
        ;;
    *)
        echo 'NEPALISH_ENGINE must be indicconformer, shruti, whisper, or neptrans.' >&2
        exit 1
        ;;
esac

for file in "$model_dir"/{config.json,model.bin,tokenizer.json}; do
    if [[ ! -x .venv/bin/wlk || ! -s "$file" ]]; then
        if [[ "$engine" == neptrans ]]; then
            echo 'Run ./setup.sh --neptrans to download NepTrans.' >&2
        else
            echo 'Run ./setup.sh before starting the demo.' >&2
        fi
        exit 1
    fi
done

export NEPALISH_ENGINE="$engine"
exec .venv/bin/python server.py \
    --host 127.0.0.1 --port "$NEPALISH_PORT" \
    --backend faster-whisper --backend-policy localagreement \
    --model "$model_name" --model_dir "$model_dir" --language "$NEPALISH_LANGUAGE" \
    --min-chunk-size 0.5 --asr-coalesce-min-s "$coalesce_seconds" --warmup-file '' \
    --log-level WARNING
