# Lumi — Audiobook Player: Full Specification

## Overview

Lumi is a self-hosted audiobook player that converts plain `.txt` books into synchronized audio with word-level highlighting. It uses AI speech synthesis (TTS) to generate audio and AI transcription (Whisper) to align every word in the text to a timestamp. The result is a browser-based reader where words highlight in real-time as the audio plays, and clicking any word seeks the audio to that point.

---

## Architecture

```
txt/*.txt
    │
    ▼ (1) TTS Service — Kokoro-82M
    │
    ├── ui/data/<book>.mp3
    │
    ▼ (2) Sync Service — faster-whisper
    │
    ├── ui/data/<book>_sync_map.json
    ├── ui/data/<book>_transcribed_words.json
    └── ui/data/<book>_segments.json
    │
    ▼ (3) rsync → TARDIS server
    │
    ▼ (4) UI Server — Node.js / Express
         Served at http://tardis:8000
```

The pipeline (`run_pipeline.sh`) orchestrates all three steps end-to-end. The UI server is a standalone Express app deployed on a remote machine called TARDIS.

---

## Components

### 1. TTS Service (`tts_service/main.py`)

**Purpose:** Convert a `.txt` book into an `.mp3` audiobook using a local neural TTS model.

**Model:** [Kokoro-82M](https://github.com/hexgrad/kokoro) — a lightweight, high-quality TTS model. Runs entirely locally.

**Inputs:**
- `--txt-file <path>` — specific `.txt` file to process (used by pipeline)
- `--txt-dir <path>` — directory of `.txt` files (default: `../txt`)
- `--output-dir <path>` — where to write `.mp3` output (default: `../ui/data`)
- `--voice <name>` — Kokoro voice (default: `af_heart`)
- `--speed <float>` — playback speed multiplier (default: `1.0`)

**Output:** `<book-name>.mp3` at 24kHz sample rate.

**Processing logic:**
1. Read and normalize whitespace in the `.txt` file.
2. Split text into chunks ≤ 200 tokens, respecting sentence boundaries (splits on `.`, `!`, `?`).
3. Run each chunk through `KPipeline` (lang `'a'` = American English).
4. Concatenate all audio arrays and write to MP3 via `soundfile`.

**Dependencies:** `kokoro`, `soundfile`, `misaki[en]`, `tqdm`, `numpy`

---

### 2. Sync Service (`sync_service/main.py`)

**Purpose:** Generate word-level timestamps for the audio by transcribing it with Whisper.

**Model:** [faster-whisper](https://github.com/guillaumekynast/faster-whisper) — quantized CTranslate2 implementation of OpenAI Whisper.

**Inputs:**
- `--file <path>` — specific `.mp3` file (used by pipeline)
- `--data-dir <path>` — directory to scan for `.mp3` files (default: `../data`)
- `--output-dir <path>` — where to write JSON outputs (default: `../ui/data`)
- `--model-size` — Whisper model variant (default: `base.en`; choices: tiny/base/small/medium/large)
- `--device` — `cuda` or `cpu` (default: `cuda`)
- `--compute-type` — quantization type (default: `int8_float16`)

**Outputs** (per book, using sanitized filename as base):
| File | Contents |
|------|----------|
| `<base>_sync_map.json` | `{ "p1": { "start": 0.0, "end": 5.78 }, ... }` — segment-level timestamps |
| `<base>_transcribed_words.json` | Array of `{ word, start, end, score }` — word-level timestamps |
| `<base>_segments.json` | Array of `{ id, text }` — clean segment text |
| `<base>.txt` | Full transcribed text joined from all segments |

**Processing logic:**
1. Get total audio duration via `ffprobe`.
2. Stream audio in 1-hour chunks using a producer-consumer pattern (background thread prefetches next chunk via `ffmpeg` while the main thread transcribes the current one).
3. Transcribe each chunk with `word_timestamps=True` and `vad_filter=True` (voice activity detection skips silence).
4. Adjust all timestamps by the chunk's time offset to produce absolute timestamps.
5. Build sync map (segment-level) and word-level data from all segments.
6. Sanitize filename (strip special characters) and write all JSON outputs.
7. Move the `.mp3` file to the output directory under the sanitized name.

**Skipping:** If `<base>_sync_map.json` already exists, the file is skipped.

**GPU memory management:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` after each file.

**Dependencies:** `faster-whisper`, `torch`, `numpy`, `tqdm`; `ffmpeg` and `ffprobe` must be on PATH.

---

### 3. Pipeline (`run_pipeline.sh`)

**Purpose:** Orchestrate TTS → Sync → Deploy in a single command.

**Usage:**
```bash
./run_pipeline.sh <path-to-txt-file>
# Example:
./run_pipeline.sh txt/"Alice in Wonderland.txt"
```

**Steps:**
1. Clears `ui/data/` and recreates it.
2. Runs TTS service → produces `<book>.mp3`.
3. Runs Sync service → produces JSON files.
4. `rsync`s the entire `ui/data/` directory to `rony@tardis:~/code/Lumi/ui/data/`.

**Environment:** Uses the project's shared `.venv` at `$PROJECT_DIR/.venv`. Prepends NVIDIA cuDNN/cuBLAS library paths to `LD_LIBRARY_PATH` if present (for GPU acceleration).

---

### 4. UI Server (`ui/server.js`)

**Purpose:** Serve the audiobook player web app and persist user settings.

**Runtime:** Node.js with Express. Managed by `pm2` on the TARDIS server.

**Port:** 8000 (fallback: 8001 if 8000 is in use).

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/books` | List available books (returns sorted array of base names with all required data files present) |
| `GET` | `/api/settings` | Get all user settings (from `settings.json`) |
| `GET` | `/api/settings/:key` | Get a specific setting value |
| `PUT` | `/api/settings/:key` | Update a specific setting value |
| `GET` | `/*` | Serve static files from `ui/` |

**Settings storage:** `ui/settings.json` — a flat JSON key-value file. Read and written synchronously on each request.

**Book discovery logic:** Scans `ui/data/` for `.mp3` files, then verifies that both `<base>_transcribed_words.json` and `<base>_sync_map.json` exist alongside each `.mp3`. Only books with all three files are returned.

**Cache headers:**
- `.mp3`, `.json` data files: `Cache-Control: public, max-age=31536000, immutable` (1 year)
- `.html`, `.js`, `.css`: `Cache-Control: public, max-age=3600` (1 hour)

**Dependencies:** `express ^4.18.2`

---

### 5. UI Client (`ui/index.html` + `ui/app.js`)

**Purpose:** Browser-based audiobook reader with real-time word highlighting and audio sync.

**Key features:**

- **Book selector** — dropdown listing all available books; remembers last selected book.
- **Audio player** — native HTML5 `<audio>` element with full playback controls.
- **Word highlighting** — current word highlighted in yellow (`#ffd700`) as audio plays.
- **Click-to-seek** — clicking any word seeks the audio to that word's timestamp.
- **Font size controls** — `+`/`-` buttons to adjust text size (range: 12–48px, step 2px).
- **Position persistence** — saves playback position every ~1 second during playback and on pause; restores on next visit.
- **Settings sync** — settings saved locally to `localStorage` immediately, then debounced (3s) and persisted to the server API.

**Rendering — chunked virtual display:**
- Paragraphs are grouped into chunks of 50 (`CHUNK_SIZE = 50`).
- Only the current chunk is rendered in the DOM at a time (prevents DOM bloat for long books).
- When playback crosses a chunk boundary, the DOM is replaced with the new chunk.
- Each word rendered as a `<span class="word" data-start="..." data-end="...">`.

**Highlight algorithm:**
1. On `timeupdate` and `seeked` events, binary-search the rendered word spans by `data-start`/`data-end`.
2. If the current time falls in a gap between words (no exact match), fall back to the last word whose start time has passed.
3. Scroll the active word into view with `scrollIntoView({ behavior: 'smooth', block: 'center' })`.

**Settings keys:**
| Key | Description |
|-----|-------------|
| `lumi_font_size` | Font size in px (default: 18) |
| `lumi_selected_book` | Last selected book base name |
| `lumi_<book>_last_time` | Last playback position (seconds) per book |

---

## Data Flow (End-to-End)

```
User provides txt/Book.txt
        │
        ▼
run_pipeline.sh
        │
        ├── tts_service/main.py
        │       Kokoro-82M (local GPU)
        │       → ui/data/Book.mp3
        │
        ├── sync_service/main.py
        │       faster-whisper (local GPU)
        │       ffmpeg/ffprobe for chunked loading
        │       → ui/data/Book_sync_map.json
        │       → ui/data/Book_transcribed_words.json
        │       → ui/data/Book_segments.json
        │
        └── rsync → rony@tardis:~/code/Lumi/ui/data/

User opens http://tardis:8000
        │
        ├── GET /api/books → list of available books
        ├── GET /api/settings → restore font size, last book, positions
        ├── GET data/Book.mp3 → audio stream
        ├── GET data/Book_transcribed_words.json → word timestamps
        └── GET data/Book_sync_map.json → segment timestamps
```

---

## Deployment

**Remote server:** `rony@tardis` (accessible as `tardis` on local network)

**URL:** `http://tardis:8000`

**UI files location:** `~/code/Lumi/ui/`

**Process manager:** `pm2`

**Deploy data after pipeline:**
```bash
rsync -av ui/data/ rony@tardis:~/code/Lumi/ui/data/
```

**Deploy UI code changes:**
```bash
ssh rony@tardis "cd ~/code/Lumi/ui && git pull && pm2 restart <name>"
```

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| TTS (speech synthesis) | Kokoro-82M (local, Python) |
| Transcription / alignment | faster-whisper / Whisper (local, Python + CUDA) |
| Audio decoding | ffmpeg / ffprobe (CLI) |
| Pipeline orchestration | Bash |
| Web server | Node.js + Express |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Settings storage | JSON file (`settings.json`) |
| Process management | pm2 |
| Deployment | rsync over SSH |

---

## Directory Structure

```
Lumi/
├── run_pipeline.sh          # End-to-end pipeline script
├── SPEC.md                  # This file
├── TARDIS.md                # Deployment reference
├── .venv/                   # Shared Python virtualenv
├── txt/                     # Input plain-text books
│   ├── Alice in Wonderland.txt
│   ├── Frankenstein Or, The Modern Prometheus.txt
│   └── Moby Dick.txt
├── tts_service/
│   ├── main.py              # Kokoro TTS runner
│   └── requirements.txt
├── sync_service/
│   ├── main.py              # Whisper sync/transcription
│   ├── requirements.txt
│   └── run_sync.sh          # Standalone sync runner
├── ui/
│   ├── index.html           # Player UI
│   ├── app.js               # Player logic
│   ├── server.js            # Express web server
│   ├── package.json
│   ├── settings.json        # Persisted user settings
│   └── data/                # Generated per-book files
│       ├── <book>.mp3
│       ├── <book>_sync_map.json
│       ├── <book>_transcribed_words.json
│       └── <book>_segments.json
└── docs/
    └── superpowers/
        └── specs/           # Design documents
```
