#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --neptrans && "$1" != --indicconformer ) ]]; then
    echo 'Usage: ./setup.sh [--neptrans | --indicconformer]' >&2
    exit 2
fi

if ! command -v uv >/dev/null; then
    echo 'Install uv and rerun ./setup.sh.' >&2
    exit 1
fi

install_environment() {
    local environment="$1"
    if [[ ! -x "$environment/bin/python" ]]; then
        uv venv --python 3.12 "$environment"
    fi
    uv pip install --python "$environment/bin/python" \
        --index pytorch-cpu=https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match --requirements requirements.txt
}

install_environment .venv

if [[ "${1:-}" == --indicconformer ]]; then
    install_environment .venv-indicconformer
    uv pip install --python .venv-indicconformer/bin/python \
        'nemo_toolkit[asr]==2.7.0' 'datasets==4.8.5' \
        'torch==2.9.1+cpu' 'torchaudio==2.9.1+cpu'
fi

CACTUS_TELEMETRY=0 HF_HUB_DOWNLOAD_TIMEOUT=60 .venv/bin/python - "${1:-}" <<'PY'
import hashlib
import sys
from pathlib import Path

from cactus import Whistle
from huggingface_hub import hf_hub_download, snapshot_download

weights_dir = Path('models/shruti-nepali')
weights_file = weights_dir / 'shruti-nepali.cact'
if not weights_file.is_file():
    hf_hub_download(
        repo_id='udgamlabs/shruti-nepali',
        filename=weights_file.name,
        revision='2ccea6014a7f1c32a7248a13d03972818252794f',
        local_dir=weights_dir,
    )
# Preload and cache Cactus's pinned native library while setup is online.
Whistle(weights_file)

model_dir = Path('models/whisper-small')
files = ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt')
if not all((model_dir / name).is_file() for name in files):
    snapshot_download(
        repo_id='Systran/faster-whisper-small',
        revision='536b0662742c02347bc0e980a01041f333bce120',
        allow_patterns=files,
        local_dir=model_dir,
    )

if sys.argv[1] == '--neptrans':
    model_dir = Path('models/neptrans-v4')
    files = ('config.json', 'model.bin', 'preprocessor_config.json', 'tokenizer.json', 'vocabulary.json',
             'LICENSE', 'NOTICE', 'OPENSLR54_LICENSE', 'WHISPER_LICENSE')
    if not all((model_dir / name).is_file() for name in files):
        snapshot_download(
            repo_id='DontHash/neptrans-v4-ct2',
            revision='7af30bbdf306e2b21096aec078d536f87c0f7eba',
            allow_patterns=files,
            local_dir=model_dir,
        )
    print('Setup complete. Launch with NEPALISH_ENGINE=neptrans ./run.sh.')
elif sys.argv[1] == '--indicconformer':
    model_dir = Path('models/indicconformer-ne-en')
    filename = 'indicconformer-ne-en-codemix.nemo'
    files = (filename, 'README.md')
    if not all((model_dir / name).is_file() for name in files):
        snapshot_download(
            repo_id='ailabpvtltd/indicconformer-nepali-english-codemix-asr',
            revision='d80adf63ca0562a5e2042f6bd4462fe0b68729a0',
            allow_patterns=files,
            local_dir=model_dir,
        )
    with (model_dir / filename).open('rb') as weights:
        if hashlib.file_digest(weights, 'sha256').hexdigest() != '6cd960445abdc986bb712995f2da647d0829ba81e2d3e7981df778ea28104c2e':
            raise SystemExit(f'Checksum mismatch for {model_dir / filename}')
    print('Setup complete. Launch with NEPALISH_ENGINE=indicconformer ./run.sh.')
else:
    print('Setup complete. Launch with ./run.sh.')
PY
