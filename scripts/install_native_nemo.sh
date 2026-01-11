#!/bin/bash
# Script to install Native AI4Bharat ASR dependencies
# Usage: ./install_native_nemo.sh

echo "📦 Installing Native AI4Bharat NeMo Dependencies..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    exit 1
fi

# Install optional group 'indic'
echo "Running: uv sync --extra indic"
uv sync --extra indic

if [ $? -eq 0 ]; then
    echo "✅ Native NeMo Installed!"
    echo "   You can now set use_native_nemo = True in config.py (Default)"
else
    echo "❌ Failed to install dependencies. Check log for conflicts."
    exit 1
fi
