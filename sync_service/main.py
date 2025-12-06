# main.py
from faster_whisper import WhisperModel
import gc
import torch
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import logging
import math
import subprocess
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_audio_duration(audio_path: str) -> float:
    """Get the duration of the audio file using ffprobe."""
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
    Load a specific chunk of audio using ffmpeg directly to numpy array.
    Reduces memory usage by not loading the full file.
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
            
        # Convert raw bytes to numpy array
        # s16le = int16
        audio_int16 = np.frombuffer(out, dtype=np.int16)
        
        # Normalize to float32 [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float32
        
    except Exception as e:
        logging.error(f"Error loading chunk {start_time}-{start_time+duration}: {e}")
        return np.array([], dtype=np.float32)

def transcribe_chunked(model: WhisperModel, audio_path: str, chunk_duration_sec=3600):
    """
    Transcribes a long audio file by streaming chunks using ffmpeg.
    Implements a Producer-Consumer pattern to prefetch the next chunk 
    while transcribing the current one.
    """
    logging.info(f"Getting duration for {audio_path}...")
    duration_sec = get_audio_duration(audio_path)
    total_chunks = math.ceil(duration_sec / chunk_duration_sec)
    logging.info(f"Audio duration: {duration_sec:.2f}s. Processing in {total_chunks} chunks.")
    
    all_segments = []
    
    # Queue for prefetching: (index, audio_data)
    # Size 1 is enough for double buffering
    prefetch_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def prefetch_worker():
        for i in range(total_chunks):
            if stop_event.is_set():
                break
                
            start_time = i * chunk_duration_sec
            # Ensure we don't request past the end, though ffmpeg handles it gracefully usually
            # But calculating exact duration helps avoids empty fetches
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
            
            segments, _ = model.transcribe(
                audio_chunk, 
                word_timestamps=True, 
                vad_filter=True
            )
            
            # Process immediately to free memory
            for segment in segments:
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
        # Ensure thread joins if we break early
        stop_event.set()
        # Empty queue to let thread finish if it's blocked on put
        while not prefetch_queue.empty():
            try:
                prefetch_queue.get_nowait()
            except queue.Empty:
                break
        loader_thread.join(timeout=5)

    return all_segments, duration_sec

def process_audiobook(model: WhisperModel, audio_path: str, text_path: str, output_dir: str):
    audio_path = Path(audio_path)
    output_base_name = audio_path.stem
    output_dir = Path(output_dir)

    sync_map_path = output_dir / f"{output_base_name}_sync_map.json"
    if sync_map_path.exists():
        logging.info(f"Output for {audio_path.name} already exists. Skipping.")
        return

    # 1. Transcribe the audio (CHUNKED STRATEGY)
    logging.info("Step 1/3: Transcribing audio with word-level timestamps...")
    
    # Use the memory-safe chunked transcriber
    raw_segments, duration = transcribe_chunked(model, str(audio_path))
    
    # Flatten word data for alignment
    word_level_data = []
    for seg in raw_segments:
        if seg.get("words"):
            word_level_data.extend(seg["words"])

    # Save the transcribed words
    output_path = output_dir / f"{output_base_name}_transcribed_words.json"
    with open(output_path, "w") as f:
        json.dump(word_level_data, f, indent=4)
    logging.info(f"Saved transcribed words to {output_path}")

    # Clean up massive audio array if it exists
    gc.collect()

    # 2. Process the ground-truth text
    logging.info("Step 2/3: Processing ground-truth text...")
    with open(text_path, "r", encoding="utf-8") as f:
        ground_truth_text = f.read()
    
    # Improved Segmentation: prevent massive blocks that crash alignment
    ground_truth_segments = []
    raw_paragraphs = ground_truth_text.split('\n\n')
    
    p_counter = 1
    for p_text in raw_paragraphs:
        cleaned = " ".join(p_text.split())
        if not cleaned: continue
        
        # Safety Check: If a paragraph is > 300 words, split it by sentences
        # This keeps the Needleman-Wunsch matrix small.
        words_in_p = cleaned.split()
        if len(words_in_p) > 300:
            # Simple sentence split approximation
            sentences = cleaned.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n').split('\n')
            for s in sentences:
                if s.strip():
                    ground_truth_segments.append({
                        "id": f"p{p_counter}",
                        "text": s.strip()
                    })
                    p_counter += 1
        else:
            ground_truth_segments.append({
                "id": f"p{p_counter}",
                "text": cleaned
            })
            p_counter += 1

    ground_truth_path = output_dir / f"{output_base_name}_segments.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_segments, f, indent=4)

    # 3. Align and Map
    logging.info("Step 3/3: Aligning transcribed text with ground-truth text...")

    sync_map = {}
    transcribed_word_cursor = 0
    
    # Optimization: Pre-calculate transcribed words for faster searching
    # We only look ahead somewhat to avoid O(N^2) on the whole book
    SEARCH_WINDOW_MULTIPLIER = 3
    MIN_SEARCH_WINDOW = 100
    MAX_SEARCH_WINDOW = 2000 # Hard cap to prevent memory spikes in NW matrix

    for segment in tqdm(ground_truth_segments, desc="Aligning segments"):
        segment_words = segment['text'].split()
        if not segment_words:
            continue

        start_pos = transcribed_word_cursor
        
        # Dynamic window size with safety cap
        window_size = max(len(segment_words) * SEARCH_WINDOW_MULTIPLIER, MIN_SEARCH_WINDOW)
        window_size = min(window_size, MAX_SEARCH_WINDOW) 
        
        end_pos = min(start_pos + window_size, len(word_level_data))
        search_window = word_level_data[start_pos:end_pos]
        
        if not search_window:
            break

        # Run memory-optimized alignment
        alignment = needleman_wunsch(segment_words, search_window)

        start_time, end_time = None, None
        last_matched_idx = -1
        
        # Create a lookup for object identity to find index
        # (Converting to list of IDs is faster than searching list of dicts)
        window_ids = [id(w) for w in search_window]

        for gt_word, whisper_word in alignment:
            if gt_word is not None and whisper_word is not None:
                if start_time is None:
                    start_time = whisper_word["start"]
                end_time = whisper_word["end"]
                
                # Find index in window efficiently
                try:
                    idx = window_ids.index(id(whisper_word))
                    last_matched_idx = idx
                except ValueError:
                    pass

        if start_time is not None and end_time is not None:
            sync_map[segment["id"]] = {
                "start": start_time,
                "end": end_time
            }
            if last_matched_idx != -1:
                transcribed_word_cursor = start_pos + last_matched_idx + 1
        else:
            transcribed_word_cursor += len(segment_words) # Fallback

    with open(sync_map_path, "w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=4)
    logging.info(f"Saved sync map to {sync_map_path}")
    logging.info(f"Processing complete. Outputs saved to {output_dir}")

def create_sync_map_from_transcription(model: WhisperModel, audio_path: str, output_dir: str):
    audio_path = Path(audio_path)
    output_base_name = audio_path.stem
    output_dir = Path(output_dir)
    text_path = audio_path.with_suffix(".txt")

    if (output_dir / f"{output_base_name}_sync_map.json").exists():
        logging.info(f"Output for {audio_path.name} already exists. Skipping.")
        return

    logging.info(f"Generating text and sync map directly from audio for {audio_path.name}...")
    
    # Use chunked transcription here too
    raw_segments, duration = transcribe_chunked(model, str(audio_path))

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

    # Save all files
    with open(output_dir / f"{output_base_name}_sync_map.json", "w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=4)
    
    with open(output_dir / f"{output_base_name}_transcribed_words.json", "w") as f:
        json.dump(word_level_data, f, indent=4)
        
    with open(output_dir / f"{output_base_name}_segments.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth_segments, f, indent=4)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(" ".join(full_text_segments))

    gc.collect()
    torch.cuda.empty_cache()

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    """
    Memory Optimized Needleman-Wunsch (Int16)
    """
    # 1. Pre-process strings
    seq1_cleaned = [word.lower().strip(".,!?\"'") for word in seq1]
    seq2_cleaned = [word['word'].lower().strip(".,!?\"'") for word in seq2]

    n = len(seq1)
    m = len(seq2)
    
    # 2. Memory Optimization: Use int16 instead of float64
    # This reduces memory footprint by 4x.
    # We use int16 because alignment scores for paragraphs won't overflow 32,767.
    score = np.zeros((n + 1, m + 1), dtype=np.int16)

    # Initialization
    # Use arange for faster vector initialization
    score[:, 0] = np.arange(n + 1) * gap_penalty
    score[0, :] = np.arange(m + 1) * gap_penalty

    # Fill table
    # Loop is unavoidable in standard python NW, but reduced dtype speeds up memory access
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = match_score if seq1_cleaned[i-1] == seq2_cleaned[j-1] else mismatch_score
            
            # Calculate candidates
            match = score[i - 1, j - 1] + match_val
            delete = score[i - 1, j] + gap_penalty
            insert = score[i, j - 1] + gap_penalty
            
            # Manual max (slightly faster than np.max for scalars)
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
        score_up = score[i][j - 1]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook synchronization tool.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing audio and text files.")
    parser.add_argument("--output-dir", type=str, default="sync_service/output", help="Directory to save output files.")
    parser.add_argument("--model-size", type=str, default="large-v2", help="Model size.")
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