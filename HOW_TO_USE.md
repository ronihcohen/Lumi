# Lumi - Audiobook Generation & Sync Guide

This guide covers the complete workflow for converting text files to audiobooks with synchronized text display.

## Overview

The pipeline consists of three main steps:

1. **TTS Service** - Convert text to audio using XTTS-v2
2. **Sync Service** - Generate text synchronization metadata using Whisper
3. **UI Deployment** - Push files to the web server

## Prerequisites

- Python 3.12
- CUDA-capable GPU (recommended)
- FFmpeg installed system-wide
- Virtual environment at `.venv/`

## 1. TTS Service (Text-to-Speech)

Converts `.txt` files from `txt/` directory into `.mp3` audiobooks in `data/`.

### Setup

```bash
cd /home/rony/code/Lumi
source .venv/bin/activate
pip install -r tts_service/requirements.txt
```

### Dependencies (requirements.txt)

```
# Core TTS - Use 0.24.2 to avoid torchcodec dependency issues
coqui-tts==0.24.2

# PyTorch ecosystem - Must use matching versions (2.5.x works well)
torch==2.5.0
torchaudio==2.5.0
torchvision==0.20.0

# Transformers - coqui-tts 0.24.2 requires <4.43.0
transformers>=4.42.0,<4.43.0

# Audio processing
pydub
soundfile>=0.12.0

# Progress display
tqdm

# Text processing (optional)
spacy
```

**Important Version Notes:**
- PyTorch 2.9.x requires `torchcodec` which has CUDA compatibility issues
- Use PyTorch 2.5.x ecosystem to avoid these problems
- coqui-tts 0.27.x requires newer transformers which conflicts with older torch

### Usage

```bash
cd tts_service
python main.py
```

**Arguments:**
- `--txt-dir` - Input directory with .txt files (default: `../txt`)
- `--output-dir` - Output directory for .mp3 files (default: `../data`)
- `--speaker-wav` - Voice reference WAV file (default: `speaker_ref.wav`)
- `--language` - Language code (default: `en`)

**Example:**
```bash
python main.py --txt-dir ../txt --output-dir ../data --speaker-wav speaker_ref.wav
```

### Input/Output

- **Input:** `txt/<book_name>.txt`
- **Output:** `data/<book_name>.mp3`

The script:
1. Splits text into ~200 character chunks (safe for XTTS-v2)
2. Synthesizes each chunk with voice cloning
3. Concatenates all chunks into a single MP3

## 2. Sync Service (Audio-Text Synchronization)

Transcribes audiobooks using Whisper and generates synchronization metadata.

### Setup

The sync service uses the same `.venv` as the main project, but requires additional dependencies:

```bash
pip install faster-whisper numpy
```

Also ensure CUDA libraries are available:
```bash
pip install nvidia-cudnn-cu13 nvidia-cublas-cu13
```

### Usage

```bash
cd sync_service
./run_sync.sh
```

Or manually:
```bash
cd sync_service
export LD_LIBRARY_PATH="$PWD/../.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/../.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
python main.py --data-dir ../data
```

**Arguments:**
- `--data-dir` - Directory containing .mp3 files (default: `../data`)
- `--output-dir` - Output directory (default: `../ui/data`)
- `--model-size` - Whisper model (default: `base.en`, options: tiny/base/small/medium/large-v3)
- `--device` - `cuda` or `cpu` (default: `cuda`)

### Input/Output

- **Input:** `data/<book_name>.mp3`
- **Output:** (all in `ui/data/`)
  - `<book_name>.mp3` - Audio file (moved from data/)
  - `<book_name>.txt` - Transcribed text
  - `<book_name>_segments.json` - Text segments with IDs
  - `<book_name>_sync_map.json` - Timing data for each segment
  - `<book_name>_transcribed_words.json` - Word-level timestamps

## 3. UI Deployment

Push processed files to the remote server.

### Server Details

- **Host:** `rony@tardis`
- **URL:** `http://tardis:8000`
- **Files location:** `~/code/Lumi/ui/`

### Deploy Data Files

```bash
rsync -av ui/data/ rony@tardis:~/code/Lumi/ui/data/
```

### Deploy UI Updates

```bash
rsync -av ui/server.py ui/app.js ui/index.html rony@tardis:~/code/Lumi/ui/
```

### Start/Restart Server

```bash
ssh rony@tardis
cd ~/code/Lumi/ui
python3 server.py
```

## Complete Workflow Example

Convert "Alice in Wonderland" from text to synchronized audiobook:

```bash
# 1. Activate environment
cd /home/rony/code/Lumi
source .venv/bin/activate

# 2. Place text file
cp "Alice in Wonderland.txt" txt/

# 3. Generate audiobook (this takes a while)
cd tts_service
python main.py

# 4. Generate sync metadata
cd ../sync_service
./run_sync.sh

# 5. Deploy to server
cd ..
rsync -av ui/data/ rony@tardis:~/code/Lumi/ui/data/
```

## Troubleshooting

### TTS Service

**Error: `torchcodec` / `libnppicc.so` not found**
- Downgrade to PyTorch 2.5.x: `pip install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0`

**Error: `transformers` import issues**
- Pin transformers: `pip install "transformers>=4.42.0,<4.43.0"`

**Error: `torchvision::nms does not exist`**
- Version mismatch - reinstall matching versions: `pip install torchvision==0.20.0`

### Sync Service

**Error: CUDNN/CuBLAS not found**
- Install NVIDIA libraries: `pip install nvidia-cudnn-cu13 nvidia-cublas-cu13`
- Or use the `run_sync.sh` script which sets `LD_LIBRARY_PATH`

**Error: FFmpeg not found**
- Install FFmpeg: `sudo apt install ffmpeg`

## Directory Structure

```
Lumi/
├── .venv/              # Python virtual environment
├── txt/                # Input text files
├── data/               # Generated audiobooks (before sync)
├── tts_service/
│   ├── main.py         # TTS conversion script
│   ├── requirements.txt
│   └── speaker_ref.wav # Voice reference file
├── sync_service/
│   ├── main.py         # Sync generation script
│   ├── run_sync.sh     # Runner with CUDA libs
│   └── requirements.txt
├── ui/
│   ├── data/           # Processed files for web UI
│   ├── server.py       # Web server
│   ├── app.js          # Frontend JS
│   └── index.html      # Frontend HTML
└── HOW_TO_USE.md       # This file
```
