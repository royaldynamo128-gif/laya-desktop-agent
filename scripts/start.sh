#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [ ! -d ".venv" ]; then
    echo "Virtual environment missing! Running setup_env.sh..."
    bash scripts/setup_env.sh
fi

source .venv/bin/activate
exec python -m app.cli run "$@"
