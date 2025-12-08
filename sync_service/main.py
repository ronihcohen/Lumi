# main.py
import argparse
import gc
import json
import logging
import math
import queue
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator, Any

import numpy as np
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def needleman_wunsch(seq1: List[str], seq2: List[Dict[str, Any]], match_score=1, mismatch_score=-1, gap_penalty=-1) -> List[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
    """
    Memory Optimized Needleman-Wunsch algorithm using Int16 for score matrix.
    Aligns a list of strings (seq1) with a list of word dictionaries (seq2).
    """
    # 1. Pre-process strings for comparison
    seq1_cleaned = [word.lower().strip(".,!?\"'") for word in seq1]
    seq2_cleaned = [word['word'].lower().strip(".,!?\"'") for word in seq2]

    n = len(seq1)
    m = len(seq2)
    
    # 2. Memory Optimization: Use int16 instead of float64/int64
    score = np.zeros((n + 1, m + 1), dtype=np.int16)

    # Initialization
    score[:, 0] = np.arange(n + 1) * gap_penalty
    score[0, :] = np.arange(m + 1) * gap_penalty

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = match_score if seq1_cleaned[i-1] == seq2_cleaned[j-1] else mismatch_score
            
            match = score[i - 1, j - 1] + match_val
            delete = score[i - 1, j] + gap_penalty
            insert = score[i, j - 1] + gap_penalty
            
            # Manual max
            if match >= delete and match >= insert:
                score[i, j] = match
            elif delete >= insert:
                score[i, j] = delete
            else:
                score[i, j] = insert

    # Traceback
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_left = score[i - 1][j]
        
        match_val = match_score if seq1_cleaned[i-1] == seq2_cleaned[j-1] else mismatch_score
        
        if score_current == score_diagonal + match_val:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append(seq1[i - 1])
            align2.append(None)
            i -= 1
        else: 
            align1.append(None)
            align2.append(seq2[j - 1])
            j -= 1
    
    while i > 0:
        align1.append(seq1[i - 1])
        align2.append(None)
        i -= 1
    while j > 0:
        align1.append(None)
        align2.append(seq2[j - 1])
        j -= 1
        
    return list(zip(reversed(align1), reversed(align2)))

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

def process_audiobook(model: WhisperModel, audio_path: str, text_path: str, output_dir: str):
    """
    Process an audiobook with existing ground-truth text.
    Transcribes audio, segments text, and aligns them.
    """
    audio_path = Path(audio_path)
    output_base_name = audio_path.stem
    output_dir = Path(output_dir)

    sync_map_path = output_dir / f"{output_base_name}_sync_map.json"
    if sync_map_path.exists():
        logging.info(f"Output for {audio_path.name} already exists. Skipping.")
        return

    # 1. Transcribe
    logging.info("Step 1/3: Transcribing audio...")
    raw_segments, _ = transcribe_audio_stream(model, str(audio_path))
    
    word_level_data = []
    for seg in raw_segments:
        if seg.get("words"):
            word_level_data.extend(seg["words"])

    # 2. Process Ground Truth
    logging.info("Step 2/3: Processing ground-truth text...")
    with open(text_path, "r", encoding="utf-8") as f:
        ground_truth_text = f.read()
    
    ground_truth_segments = []
    raw_paragraphs = ground_truth_text.split('\n\n')
    
    p_counter = 1
    for p_text in raw_paragraphs:
        cleaned = " ".join(p_text.split())
        if not cleaned: continue
        
        words_in_p = cleaned.split()
        if len(words_in_p) > 300:
            # Simple sentence split for large paragraphs
            sentences = cleaned.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n').split('\n')
            for s in sentences:
                if s.strip():
                    ground_truth_segments.append({"id": f"p{p_counter}", "text": s.strip()})
                    p_counter += 1
        else:
            ground_truth_segments.append({"id": f"p{p_counter}", "text": cleaned})
            p_counter += 1

    # 3. Align
    logging.info("Step 3/3: Aligning...")
    sync_map = {}
    transcribed_word_cursor = 0
    
    # Constants for Windowed Search
    SEARCH_WINDOW_MULTIPLIER = 3
    MIN_SEARCH_WINDOW = 100
    MAX_SEARCH_WINDOW = 2000 

    for segment in tqdm(ground_truth_segments, desc="Aligning segments"):
        segment_words = segment['text'].split()
        if not segment_words:
            continue

        start_pos = transcribed_word_cursor
        window_size = max(len(segment_words) * SEARCH_WINDOW_MULTIPLIER, MIN_SEARCH_WINDOW)
        window_size = min(window_size, MAX_SEARCH_WINDOW) 
        
        end_pos = min(start_pos + window_size, len(word_level_data))
        search_window = word_level_data[start_pos:end_pos]
        
        if not search_window:
            break

        alignment = needleman_wunsch(segment_words, search_window)

        start_time, end_time = None, None
        last_matched_idx = -1
        
        window_ids = [id(w) for w in search_window]

        for gt_word, whisper_word in alignment:
            if gt_word is not None and whisper_word is not None:
                if start_time is None:
                    start_time = whisper_word["start"]
                end_time = whisper_word["end"]
                
                try:
                    idx = window_ids.index(id(whisper_word))
                    last_matched_idx = idx
                except ValueError:
                    pass

        if start_time is not None and end_time is not None:
            sync_map[segment["id"]] = {"start": start_time, "end": end_time}
            if last_matched_idx != -1:
                transcribed_word_cursor = start_pos + last_matched_idx + 1
        else:
            transcribed_word_cursor += len(segment_words)

    # Save
    save_sync_outputs(output_dir, output_base_name, sync_map, word_level_data, ground_truth_segments)

def create_sync_map_from_transcription(model: WhisperModel, audio_path: str, output_dir: str):
    """
    Generate sync map directly from audio transcription (no ground truth).
    """
    audio_path = Path(audio_path)
    output_base_name = audio_path.stem
    output_dir = Path(output_dir)

    if (output_dir / f"{output_base_name}_sync_map.json").exists():
        logging.info(f"Output for {audio_path.name} already exists. Skipping.")
        return

    logging.info(f"Generating full transcription for {audio_path.name}...")
    
    try:
        raw_segments, _ = transcribe_audio_stream(model, str(audio_path))
    except Exception as e:
        logging.error(f"Failed to transcribe {audio_path}: {e}")
        return

    sync_map = {}
    word_level_data = []
    full_text_segments = []
    ground_truth_segments = []

    for i, seg in enumerate(raw_segments):
        seg_id = f"p{i+1}"
        sync_map[seg_id] = {
            "start": seg["start"],
            "end": seg["end"]
        }
        
        clean_text = " ".join(seg["text"].split())
        if clean_text:
            full_text_segments.append(seg["text"])
            ground_truth_segments.append({
                "id": seg_id,
                "text": clean_text
            })
        
        if seg.get("words"):
            word_level_data.extend(seg["words"])

    full_text = " ".join(full_text_segments)
    save_sync_outputs(output_dir, output_base_name, sync_map, word_level_data, ground_truth_segments, full_text)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook synchronization tool.")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing audio and text files.")
    parser.add_argument("--output-dir", type=str, default="sync_service/output", help="Directory to save output files.")
    parser.add_argument("--model-size", type=str, default="tiny", help="Model size.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
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
                    text_file = audio_file.with_suffix(".txt")

                    if text_file.exists():
                        process_audiobook(model, str(audio_file), str(text_file), str(output_dir))
                    else:
                        create_sync_map_from_transcription(model, str(audio_file), str(output_dir))
        except Exception as e:
            logging.critical(f"Fatal error initializing model or processing: {e}")