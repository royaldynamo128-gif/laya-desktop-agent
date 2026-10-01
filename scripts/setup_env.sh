#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Setting up virtual environment in ${PROJECT_ROOT}/.venv ==="
cd "${PROJECT_ROOT}"

# Prefer uv (fast), fall back to python3 -m venv + pip
if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    UV=""
fi

if [ ! -d ".venv" ]; then
    if [ -n "$UV" ]; then
        "$UV" venv --python 3.12 .venv
    else
        python3.12 -m venv .venv
    fi
fi

if [ -n "$UV" ]; then
    "$UV" pip install --python .venv/bin/python "setuptools<81"
    "$UV" pip install --python .venv/bin/python -e .
else
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install "setuptools<81"
    .venv/bin/python -m pip install -e .
fi

echo "=== Verifying core imports ==="
.venv/bin/python - << 'EOF'
required = ["psutil", "pydantic", "yaml", "rich", "click", "numpy", "sounddevice", "webrtcvad"]
optional = {
    "faster_whisper": "STT (required for voice input)",
    "piper": "TTS (optional; agent works without voice replies)",
    "evdev": "global Win+Alt shortcut (required for push-to-talk)",
    "pynvml": "GPU monitoring (optional)",
    "laya": "neural router (optional; semantic-parser fallback is automatic)",
    "PIL": "image conversion for save-as (optional)",
    "playwright": "browser DOM automation (optional; xdg-open fallback is automatic)",
}
failed_required = []
for mod in required:
    try:
        __import__(mod)
    except Exception as e:
        failed_required.append(f"{mod}: {e}")
if failed_required:
    print("MISSING REQUIRED:", "; ".join(failed_required))
    raise SystemExit(1)
for mod, why in optional.items():
    try:
        __import__(mod)
        print(f"  ok      {mod}")
    except Exception:
        print(f"  MISSING {mod} — {why} (agent degrades gracefully)")
print("All required dependencies imported successfully!")
EOF

echo "=== Setup complete ==="
echo "Start the daemon with: bash scripts/start.sh"
echo "Run the test suite with: bash scripts/test.sh"
echo "Diagnose your environment with: .venv/bin/python -m app.cli doctor"
