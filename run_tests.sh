#!/bin/bash
set -e

NUKE=false
for arg in "$@"; do
  if [ "$arg" == "--nuke" ]; then
    NUKE=true
    break
  fi
done

if [ "$NUKE" = true ]; then
    REPLY="y"
else
    read -p "Nuke venv for reproducibility check? (y/n) " -n 1 -r
    echo
fi

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d ".venv" ]; then
        echo "Removing .venv..."
        rm -rf .venv
    fi
    echo "Repulling dependencies..."
    uv sync
fi

# Filter out --nuke from args passed to python script
ARGS=()
for arg in "$@"; do
  if [ "$arg" != "--nuke" ]; then
    ARGS+=("$arg")
  fi
done

# Use uv run to ensure environment consistency
uv run python scripts/run_tests.py "${ARGS[@]}"
