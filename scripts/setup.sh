#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
CLONE_DIR="$EXTERNAL_DIR/ComptonMatrixExact"

if [ -d "$CLONE_DIR" ]; then
    echo "MACE already cloned at $CLONE_DIR"
else
    echo "Cloning MACE into $CLONE_DIR ..."
    mkdir -p "$EXTERNAL_DIR"
    git clone git@github.com:ItamarShmelo/MACE.git "$CLONE_DIR"
fi

echo "Building MACE ..."
cd "$CLONE_DIR"
uv sync
uv pip install -e .

echo "Creating output directories ..."
mkdir -p "$REPO_ROOT/docs/data"
mkdir -p "$REPO_ROOT/output/mc_tables"
mkdir -p "$REPO_ROOT/logs"

echo "Setup complete."
echo "  Solver venv: $CLONE_DIR/.venv/bin/python3"
echo "  Test with:   $CLONE_DIR/.venv/bin/python3 -c 'import compton_matrix; print(\"OK\")'"
