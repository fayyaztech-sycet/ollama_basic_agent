#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
#  setup.sh — one-time environment setup for AI CLI Agent
# ─────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo ""
echo "╔══════════════════════════════╗"
echo "║   AI CLI Agent — Setup       ║"
echo "╚══════════════════════════════╝"
echo ""

# ── 1. Python check ──────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[error] python3 not found. Install it and rerun."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/4] Python $PYTHON_VERSION found."

# ── 2. Virtual environment ───────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Virtual environment already exists at .venv — skipping creation."
else
    echo "[2/4] Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
    echo "      Done."
fi

# ── 3. Install dependencies ──────────────────────────
echo "[3/4] Installing dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_DIR/requirements.txt"
echo "      Done."

# ── 4. .env setup ────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "[4/4] .env already exists — skipping copy."
else
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "[4/4] Created .env from .env.example."
    echo "      Edit .env to customise your configuration."
fi

echo ""
echo "Setup complete!  Run the agent with:"
echo "  ./run.sh"
echo ""
