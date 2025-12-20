
PROJECT_ROOT=$(pwd)
VENV_DIR=".venv"
DLIB_SOURCE_DIR="/tmp/dlib_build" # Use /tmp/ for clean separation
DLIB_VERSION="v19.24.6"
CUDA_PATH="/usr/local/cuda"


full_cleanup_dlib() {
    echo "## Performing Full Cleanup (Removing installed package and source clone)..."
    
    # Deactivate venv if active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate 2> /dev/null
    fi

    # Remove installed dlib package from venv
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate" 2> /dev/null
        uv pip uninstall dlib -y 2> /dev/null
        deactivate 2> /dev/null
    fi

    # Remove dlib source clone
    if [ -d "$DLIB_SOURCE_DIR" ]; then
        echo "Removing dlib source directory: $DLIB_SOURCE_DIR"
        rm -rf "$DLIB_SOURCE_DIR"
    fi

    if [ -f "t.py" ]; then
        rm "t.py"
    fi

    echo "Full Cleanup finished."
}

setup_and_build_dlib() {
    echo "## 1. Auto-Cleanup and Project Setup..."
    
    if [ -f "t.py" ]; then rm "t.py"; fi
    if [ -d "$DLIB_SOURCE_DIR/build" ]; then rm -rf "$DLIB_SOURCE_DIR/build"; fi
    
    cd "$PROJECT_ROOT"

    # Create Venv
    if [ ! -d "$VENV_DIR" ]; then
        uv venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    # Clean uv.lock
    if [ -f "uv.lock" ]; then rm "uv.lock"; fi
    uv lock > /dev/null
    if grep -q "dlib" "uv.lock"; then
        echo "WARNING: 'dlib' still found in uv.lock. Check pyproject.toml!"
    fi

    echo "## 2. Cloning and Configuring dlib Source..."
    if [ -d "$DLIB_SOURCE_DIR" ]; then
        cd "$DLIB_SOURCE_DIR"
        git checkout "$DLIB_VERSION"
    else
        git clone https://github.com/davisking/dlib.git "$DLIB_SOURCE_DIR"
        cd "$DLIB_SOURCE_DIR"
        git checkout "$DLIB_VERSION"
    fi

    echo "## 3. Setting CUDA Environment Variables and Installing..."
    
    export CUDA_PATH="$CUDA_PATH"
    export DLIB_USE_CUDA=1
    export DLIB_USE_CUBLAS=1

    # uv pip handles the build using CMake's find_package(CUDA)
    uv pip install . --no-build-isolation --no-binary dlib

    echo "## 4. Verification..."
    cd "$PROJECT_ROOT"

    # Create verification script t.py
    cat << EOF > t.py
import dlib
print("version:", dlib.__version__)
print("CUDA enabled:", dlib.DLIB_USE_CUDA)
print("Devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else "No CUDA")
EOF

    python t.py

    echo "Build Process Complete (Check output for 'CUDA enabled: True')"
    deactivate
}

# Main Script Execution

if [[ "$1" == "--cleanup" ]]; then
    full_cleanup_dlib
elif [[ "$1" == "--build" || -z "$1" ]]; then
    setup_and_build_dlib
else
    echo "Usage: $0 [ --build | --cleanup ]"
    echo "Default action is --build."
fi