# main.py
from faster_whisper import WhisperModel
import gc
import torch
import os
import json
from pathlib import Path
import pydub
import numpy as np
import argparse

def process_audiobook(audio_path: str, text_path: str, output_dir: str):
    """
    Processes an audiobook and its text to generate a synchronization map.

    Args:
        audio_path (str): Path to the audiobook file.
        text_path (str): Path to the text file.
        output_dir (str): Directory to save the output files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    compute_type = "int8" if device == "cpu" else "float16"

    # 1. Load and transcribe the audio
    print("Loading Whisper model...")
    model = WhisperModel("large-v2", device=device, compute_type=compute_type)
    
    print("Transcribing audio with word-level timestamps...")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    # Save the transcribed words
    output_path = Path(output_dir) / "transcribed_words.json"
    
    # The segments object is a generator, so we need to process it
    word_level_data = []
    for segment in segments:
        for word in segment.words:
            word_level_data.append({
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "score": word.probability
            })

    with open(output_path, "w") as f:
        json.dump(word_level_data, f, indent=4)

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Process the ground-truth text
    print("Processing ground-truth text...")
    with open(text_path, "r", encoding="utf-8") as f:
        ground_truth_text = f.read()
    
    # Simple paragraph segmentation (split by double newline)
    ground_truth_segments_raw = ground_truth_text.split('\\n\\n')
    ground_truth_segments = []
    for i, seg_text in enumerate(ground_truth_segments_raw):
        # Normalize whitespace and filter out empty segments
        cleaned_text = " ".join(seg_text.split())
        if cleaned_text:
            ground_truth_segments.append({
                "id": f"p{i+1}",
                "text": cleaned_text
            })

    # Save the segmented ground-truth text
    ground_truth_path = Path(output_dir) / "segments.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_segments, f, indent=4)

    # 3. Align and Map
    print("Aligning transcribed text with ground-truth text...")
    
    sync_map = {}
    
    # Get all words from the ground-truth segments
    all_ground_truth_words = [word for segment in ground_truth_segments for word in segment['text'].split()]
    
    # Perform alignment
    alignment = needleman_wunsch(all_ground_truth_words, word_level_data)

    # Create a map from ground-truth words to timestamps
    word_to_time = {}
    current_word_index = 0
    for gt_word, whisper_word_data in alignment:
        if gt_word is not None and whisper_word_data is not None:
            word_to_time[current_word_index] = {
                "start": whisper_word_data["start"],
                "end": whisper_word_data["end"]
            }
        if gt_word is not None:
            current_word_index += 1

    # 4. Generate the final sync map
    print("Generating final sync map...")
    current_word_index = 0
    for segment in ground_truth_segments:
        segment_words = segment['text'].split()
        num_segment_words = len(segment_words)
        
        start_time, end_time = None, None
        
        # Find the start time
        for i in range(num_segment_words):
            if current_word_index + i in word_to_time:
                start_time = word_to_time[current_word_index + i]["start"]
                break
        
        # Find the end time
        for i in range(num_segment_words - 1, -1, -1):
            if current_word_index + i in word_to_time:
                end_time = word_to_time[current_word_index + i]["end"]
                break

        if start_time is not None and end_time is not None:
            sync_map[segment["id"]] = {
                "start": start_time,
                "end": end_time
            }
        
        current_word_index += num_segment_words

    # Save the final sync map
    sync_map_path = Path(output_dir) / "sync_map.json"
    with open(sync_map_path, "w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=4)

    print(f"Processing complete. Outputs saved to {output_dir}")

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
    """
    Performs Needleman-Wunsch alignment on two sequences.

    Args:
        seq1 (list): The first sequence.
        seq2 (list): The second sequence.
        match_score (int): The score for a match.
        mismatch_score (int): The score for a mismatch.
        gap_penalty (int): The penalty for a gap.

    Returns:
        list: A list of tuples representing the alignment.
    """
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n + 1, m + 1))

    # Initialization
    for i in range(n + 1):
        score[i][0] = gap_penalty * i
    for j in range(m + 1):
        score[0][j] = gap_penalty * j

    # Fill the score matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + (match_score if seq1[i - 1].lower() == seq2[j - 1]['word'].lower().strip(".,!?\"'") else mismatch_score)
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    # Traceback
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_up = score[i][j - 1]
        score_left = score[i - 1][j]

        if score_current == score_diagonal + (match_score if seq1[i - 1].lower() == seq2[j - 1]['word'].lower().strip(".,!?\"'") else mismatch_score):
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append(seq1[i - 1])
            align2.append(None)
            i -= 1
        else: # score_current == score_up + gap_penalty
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

def main(audio_path, text_path, output_dir):
    """
    Main function to run the synchronization process.
    """
    process_audiobook(audio_path, text_path, output_dir)

if __name__ == "__main__":
    test_data_dir = Path("c:/Code/Lumi/data")
    output_dir = Path("c:/Code/Lumi/sync_service/output")

    if not test_data_dir.exists():
        print(f"Error: Directory not found at '{test_data_dir}'")
    else:
        # Find the first .mp3 and .txt pair
        audio_file = next(test_data_dir.glob("*.mp3"), None)
        text_file = next(test_data_dir.glob("*.txt"), None)

        if audio_file and text_file:
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            main(str(audio_file), str(text_file), str(output_dir))
        else:
            print("No matching .mp3 and .txt file pair found in the test_data directory.")
