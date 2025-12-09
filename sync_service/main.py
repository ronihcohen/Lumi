# main.py
import argparse
import gc
import json
import logging
import math
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by replacing special characters with underscores.
    Removes characters that cause issues in URLs and file systems.
    
    Args:
        filename: The original filename (without extension).
        
    Returns:
        Sanitized filename safe for URLs and file systems.
    """
    # Replace common problematic characters with underscores
    # Includes: ; , : ? * " < > | ' & # % @ ! $ ^ + = ` ~ [ ] { } ( )
    sanitized = re.sub(r'[;,:\?\*"<>\|\'\&#%@!\$\^\+=`~\[\]\{\}\(\)]', '_', filename)
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of the audio file in seconds using ffprobe.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Duration in seconds as a float.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(output)
    except Exception as e:
        logging.error(f"Failed to get duration with ffprobe: {e}")
        raise

def load_audio_chunk(audio_path: str, start_time: float, duration: float, sr=16000) -> np.ndarray:
    """
    Load a specific chunk of audio using ffmpeg directly to a numpy array.
    Reduces memory usage by not loading the full file.
    
    Args:
        audio_path: Path to the audio file.
        start_time: Start time in seconds.
        duration: Duration to read in seconds.
        sr: Sample rate (default 16000 for Whisper).
        
    Returns:
        Numpy array of float32 audio samples.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sr),
        "-"
    ]
    
    try:
        # Run ffmpeg and capture stdout
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL
        )
        out, _ = process.communicate()
        
        if not out:
            return np.array([], dtype=np.float32)
            
        # Convert raw bytes to numpy array (s16le = int16)
        audio_int16 = np.frombuffer(out, dtype=np.int16)
        
        # Normalize to float32 [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float32
        
    except Exception as e:
        logging.error(f"Error loading chunk {start_time}-{start_time+duration}: {e}")
        return np.array([], dtype=np.float32)

def transcribe_audio_stream(model: WhisperModel, audio_path: str, chunk_duration_sec: int = 3600) -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcribes a long audio file by streaming chunks using ffmpeg.
    Implements a Producer-Consumer pattern to prefetch the next chunk 
    while transcribing the current one.
    
    Args:
        model: Loaded WhisperModel instance.
        audio_path: Path to the audio file.
        chunk_duration_sec: Length of each chunk in seconds (default 1 hour).
        
    Returns:
        Tuple of (all_segments_data, total_duration_sec).
    """
    logging.info(f"Getting duration for {audio_path}...")
    duration_sec = get_audio_duration(audio_path)
    total_chunks = math.ceil(duration_sec / chunk_duration_sec)
    logging.info(f"Audio duration: {duration_sec:.2f}s. Processing in {total_chunks} chunks.")
    
    all_segments = []
    
    # Queue for prefetching: (index, audio_data, time_offset)
    # Size 1 is enough for double buffering
    prefetch_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def prefetch_worker():
        """Background thread to load audio chunks."""
        for i in range(total_chunks):
            if stop_event.is_set():
                break
                
            start_time = i * chunk_duration_sec
            # Calculate exact duration to strictly avoid reading past end
            dur = min(chunk_duration_sec, duration_sec - start_time)
            
            logging.info(f"Prefetching chunk {i+1}/{total_chunks}...")
            audio_data = load_audio_chunk(audio_path, start_time, dur)
            
            prefetch_queue.put((i, audio_data, start_time))
            
        prefetch_queue.put(None) # Signal done

    # Start prefetcher
    loader_thread = threading.Thread(target=prefetch_worker, daemon=True)
    loader_thread.start()

    try:
        while True:
            item = prefetch_queue.get()
            if item is None:
                break
                
            i, audio_chunk, time_offset = item
            
            if len(audio_chunk) == 0:
                logging.warning(f"Chunk {i+1} was empty. Skipping.")
                continue

            logging.info(f"Transcribing chunk {i+1}/{total_chunks} (Offset: {time_offset}s, Size: {len(audio_chunk)} samples)...")
            
            segments_gen, _ = model.transcribe(
                audio_chunk, 
                word_timestamps=True, 
                vad_filter=True
            )
            
            # Process generator immediately to free memory
            for segment in segments_gen:
                segment_start = segment.start + time_offset
                segment_end = segment.end + time_offset
                
                adjusted_words = []
                if segment.words:
                    for word in segment.words:
                        adjusted_words.append({
                            "word": word.word,
                            "start": word.start + time_offset,
                            "end": word.end + time_offset,
                            "score": word.probability
                        })
                
                all_segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment.text,
                    "words": adjusted_words
                })
                
            # Explicit cleanup
            del audio_chunk
            gc.collect()
            
    except KeyboardInterrupt:
        logging.info("Transcription interrupted. Stopping prefetcher...")
        stop_event.set()
        raise
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        stop_event.set()
        raise
    finally:
        stop_event.set()
        # Empty queue to let thread finish if it's blocked on put
        while not prefetch_queue.empty():
            try:
                prefetch_queue.get_nowait()
            except queue.Empty:
                break
        loader_thread.join(timeout=5)

    return all_segments, duration_sec

def save_sync_outputs(output_dir: Path, base_name: str, sync_map: Dict, word_data: List[Dict], segments: List[Dict], text_content: Optional[str] = None):
    """Helper to save all output JSON files."""
    
    with open(output_dir / f"{base_name}_sync_map.json", "w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=4)
        
    with open(output_dir / f"{base_name}_transcribed_words.json", "w") as f:
        json.dump(word_data, f, indent=4) # No utf-8 needed usually for pure ascii, but safe to default
        
    with open(output_dir / f"{base_name}_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4)

    if text_content:
        with open(output_dir / f"{base_name}.txt", "w", encoding="utf-8") as f:
            f.write(text_content)
            
    logging.info(f"Saved outputs to {output_dir}")

def create_sync_map_from_transcription(model: WhisperModel, audio_path: str, output_dir: str) -> str:
    """
    Generate sync map directly from audio transcription.
    
    Returns:
        The sanitized base name used for output files.
    """
    audio_path = Path(audio_path)
    output_base_name = sanitize_filename(audio_path.stem)
    output_dir = Path(output_dir)

    if (output_dir / f"{output_base_name}_sync_map.json").exists():
        logging.info(f"Output for {audio_path.name} already exists (as {output_base_name}). Skipping.")
        return output_base_name

    logging.info(f"Generating full transcription for {audio_path.name}...")
    
    try:
        raw_segments, _ = transcribe_audio_stream(model, str(audio_path))
    except Exception as e:
        logging.error(f"Failed to transcribe {audio_path}: {e}")
        return

    sync_map = {}
    word_level_data = []
    full_text_segments = []
    segments = []

    for i, seg in enumerate(raw_segments):
        seg_id = f"p{i+1}"
        sync_map[seg_id] = {
            "start": seg["start"],
            "end": seg["end"]
        }
        
        clean_text = " ".join(seg["text"].split())
        if clean_text:
            full_text_segments.append(seg["text"])
            segments.append({
                "id": seg_id,
                "text": clean_text
            })
        
        if seg.get("words"):
            word_level_data.extend(seg["words"])

    full_text = " ".join(full_text_segments)
    save_sync_outputs(output_dir, output_base_name, sync_map, word_level_data, segments, full_text)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook synchronization tool.")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing audio files.")
    parser.add_argument("--output-dir", type=str, default="../ui/data", help="Directory to save output files.")
    parser.add_argument("--model-size", type=str, default="base.en", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                                 "medium", "medium.en", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model size (tiny/base/small=faster, medium/large=more accurate).")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu').")
    parser.add_argument("--compute-type", type=str, default="int8_float16", help="Compute type.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logging.error(f"Data directory not found at '{data_dir}'")
    else:
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Whisper model '{args.model_size}' on {device}...")
        
        try:
            model = WhisperModel(args.model_size, device=device, compute_type=args.compute_type)
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_files = list(data_dir.glob("*.mp3"))
            if not audio_files:
                logging.warning(f"No .mp3 files found in {data_dir}.")
            else:
                logging.info(f"Found {len(audio_files)} MP3 file(s).")
                for audio_file in audio_files:
                    logging.info(f"--- Processing: {audio_file.name} ---")
                    sanitized_name = create_sync_map_from_transcription(model, str(audio_file), str(output_dir))
                    
                    # Move processed input file to output folder with sanitized name
                    sanitized_audio_name = f"{sanitized_name}.mp3"
                    shutil.move(str(audio_file), str(output_dir / sanitized_audio_name))
                    logging.info(f"Moved {audio_file.name} -> {sanitized_audio_name} to {output_dir}")
        except Exception as e:
            logging.critical(f"Fatal error initializing model or processing: {e}")