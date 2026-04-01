#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
#  run.sh — activate venv and launch AI CLI Agent
# ─────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# ── Guard: setup required ────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[error] Virtual environment not found. Run setup first:"
    echo "  ./setup.sh"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "[warning] .env not found — using default values."
    echo "          Run './setup.sh' to generate one from .env.example."
fi

# ── Activate & launch ────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

exec python "$PROJECT_DIR/agent_cli.py"
