# Lumi - Quick Command Reference

## Setup (One-Time)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r sync_service\requirements.txt
```

---

## Process Audiobook (Sync Service)

### Quick Start Commands

```powershell
# Fastest (Tiny Model - Testing)
python sync_service\main.py --model-size tiny

# Recommended (Base Model - Good Balance)
python sync_service\main.py --model-size base

# High Accuracy (Large Model - Slow)
python sync_service\main.py --model-size large-v3
```

### Advanced Options

```powershell
# CPU Only (No GPU)
python sync_service\main.py --model-size base --device cpu

# Custom Directories
python sync_service\main.py --data-dir "path\to\audiobooks" --output-dir "path\to\output"

# Maximum Accuracy (Requires GPU)
python sync_service\main.py --model-size large-v3 --device cuda --compute-type float16
```

### Model Size Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick testing |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended** |
| `small` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good balance |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | Professional use |

---

## Copy Files to UI

```powershell
# Copy all generated files
copy sync_service\output\*.json ui\data\
copy data\*.mp3 ui\data\
```

---

## Run UI Server

```powershell
# Start server (default port 8000)
python ui\server.py
```

Then open browser to: **http://localhost:8000**

---

## Complete Workflow (New Audiobook)

```powershell
# 1. Place files in data folder
#    - my_book.mp3
#    - my_book.txt (optional)

# 2. Activate environment
.\.venv\Scripts\Activate.ps1

# 3. Process audiobook
python sync_service\main.py --model-size base

# 4. Copy to UI
copy sync_service\output\*.json ui\data\
copy data\*.mp3 ui\data\

# 5. Start server
python ui\server.py

# 6. Open http://localhost:8000 in browser
```

---

## Troubleshooting

### Command: Check ffmpeg is installed
```powershell
ffmpeg -version
ffprobe -version
```

### Command: Test server is running
```powershell
curl http://localhost:8000/api/books
```

### Command: Check CUDA availability
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue: Port already in use
The server auto-switches to port 8001. Check console output for the actual port.

---

## File Requirements

For a book to appear in the UI, you need in `ui\data\`:
- ✅ `{book}.mp3`
- ✅ `{book}_sync_map.json`
- ✅ `{book}_transcribed_words.json`

(The `_segments.json` file is optional for the UI)

---

## Model Download Locations

Models auto-download to:
- **Windows**: `C:\Users\{username}\.cache\huggingface\hub\`
- **Linux/Mac**: `~/.cache/huggingface/hub/`

First run will download ~1-3 GB depending on model size.

---

## Performance Estimates

**10-hour audiobook on RTX 3080:**
- `tiny`: ~30 minutes
- `base`: ~1 hour  ⭐ **Recommended**
- `small`: ~2 hours
- `medium`: ~4 hours
- `large-v3`: ~8 hours

**CPU-only** (Intel i7): 10-50x slower than GPU

---

## Quick Links

- **Main Documentation**: [`LUMI_DESIGN_DOCUMENTATION.md`](LUMI_DESIGN_DOCUMENTATION.md)
- **Original Design**: [`audiobook_to_text_sync_app_design.md`](audiobook_to_text_sync_app_design.md)
