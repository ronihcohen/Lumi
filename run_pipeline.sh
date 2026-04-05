#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/.venv"

CUDNN_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib"

if [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:$LD_LIBRARY_PATH"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <txt-file>"
    echo "Example: $0 txt/Alice\ in\ Wonderland.txt"
    exit 1
fi

TXT_FILE="$1"
if [ ! -f "$TXT_FILE" ]; then
    echo "Error: File not found: $TXT_FILE"
    exit 1
fi

BASENAME=$(basename "$TXT_FILE" .txt)
OUTPUT_DIR="$PROJECT_DIR/ui/data"
mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Running TTS service ==="
"$VENV_DIR/bin/python" "$PROJECT_DIR/tts_service/main.py" --txt-file "$TXT_FILE" --output-dir "$OUTPUT_DIR"

MP3_FILE="$OUTPUT_DIR/${BASENAME}.mp3"
if [ ! -f "$MP3_FILE" ]; then
    echo "Error: TTS did not generate $MP3_FILE"
    exit 1
fi

echo ""
echo "=== Step 2: Running Sync service ==="
"$VENV_DIR/bin/python" "$PROJECT_DIR/sync_service/main.py" --file "$OUTPUT_DIR/${BASENAME}.mp3" --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 3: Pushing data to TARDIS server ==="
rsync -av "$OUTPUT_DIR/" rony@tardis:~/code/Lumi/ui/data/

echo ""
echo "=== Complete ==="
echo "Outputs in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"
