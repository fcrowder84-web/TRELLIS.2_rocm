#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRELLIS_PYTHON:-/home/foster/trellis2-env/bin/python}"
PORT="${TRELLIS_STUDIO_PORT:-7860}"
cd "$REPO"
export TRELLIS_STUDIO_PORT="$PORT"
export PYTHONUNBUFFERED=1
exec "$PYTHON" "$REPO/tools/spirit_legacy_gui.py"
