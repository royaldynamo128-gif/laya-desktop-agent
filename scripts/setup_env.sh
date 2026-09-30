#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Setting up virtual environment in ${PROJECT_ROOT}/.venv ==="
cd "${PROJECT_ROOT}"

if [ ! -d ".venv" ]; then
    /home/rai/.local/bin/uv venv --python 3.12 .venv
fi

echo "=== Installing dependencies via uv pip ==="
/home/rai/.local/bin/uv pip install --python .venv/bin/python "setuptools<70"
/home/rai/.local/bin/uv pip install --python .venv/bin/python -e .

echo "=== Verifying core imports ==="
.venv/bin/python -c "
import pynvml
import psutil
import sounddevice
import webrtcvad
import faster_whisper
import piper
import laya
import pydantic
import yaml
import rich
print('All core dependencies imported successfully!')
"

echo "=== Setup complete ==="
