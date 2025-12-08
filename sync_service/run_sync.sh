#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR/.."
VENV_DIR="$PROJECT_DIR/.venv"

# Define library paths for CUDNN 9 (provided by nvidia-cudnn-cu13) and CuBLAS
CUDNN_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib"

# Check if directories exist
if [ ! -d "$CUDNN_LIB" ]; then
    echo "Error: CUDNN lib directory not found at $CUDNN_LIB"
    echo "Please ensure dependencies are installed: pip install -r sync_service/requirements.txt"
    exit 1
fi

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:$LD_LIBRARY_PATH"

echo "Running sync service..."
echo "Using python: $VENV_DIR/bin/python"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

"$VENV_DIR/bin/python" "$PROJECT_DIR/sync_service/main.py" --data-dir "$PROJECT_DIR/data"
