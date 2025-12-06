# main.py
from faster_whisper import WhisperModel
import gc
import torch
import json
from pathlib import Path
import numpy as np

def process_audiobook(model: WhisperModel, audio_path: str, text_path: str, output_dir: str):
    """
    Processes an audiobook and its text to generate a synchronization map.

    Args:
        model (WhisperModel): The loaded Whisper model.
        audio_path (str): Path to the audiobook file.
        text_path (str): Path to the text file.
        output_dir (str): Directory to save the output files.
    """
    # 1. Transcribe the audio
    print("Transcribing audio with word-level timestamps...")
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    
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

    # 3. Align and Map (Segment by Segment)
    print("Aligning transcribed text with ground-truth text (segment by segment)...")
    from tqdm import tqdm # Import tqdm for progress bar

    sync_map = {}
    transcribed_word_cursor = 0
    SEARCH_WINDOW_MULTIPLIER = 3 # Look at 3x the number of ground-truth words
    MIN_SEARCH_WINDOW = 100 # With a minimum of 100 words

    for segment in tqdm(ground_truth_segments):
        segment_words = segment['text'].split()
        if not segment_words:
            continue

        # Define a search window in the transcribed text to find a match
        start_pos = transcribed_word_cursor
        window_size = max(len(segment_words) * SEARCH_WINDOW_MULTIPLIER, MIN_SEARCH_WINDOW)
        end_pos = min(start_pos + window_size, len(word_level_data))
        
        search_window = word_level_data[start_pos:end_pos]
        
        if not search_window:
            print("No more transcribed words to process. Stopping alignment.")
            break

        # Perform alignment on the smaller window
        alignment = needleman_wunsch(segment_words, search_window)

        # Extract timestamps and find how many transcribed words were consumed
        start_time, end_time = None, None
        last_matched_word_index_in_window = -1

        # To get the index of the word object in the window
        word_to_index_in_window = {id(word): i for i, word in enumerate(search_window)}

        for gt_word, whisper_word in alignment:
            if gt_word is not None and whisper_word is not None:
                if start_time is None: # First match
                    start_time = whisper_word["start"]
                
                # Keep updating end_time and the last matched index
                end_time = whisper_word["end"]
                last_matched_word_index_in_window = word_to_index_in_window.get(id(whisper_word), -1)

        if start_time is not None and end_time is not None:
            sync_map[segment["id"]] = {
                "start": start_time,
                "end": end_time
            }
            
            # Move the cursor forward to the word after the last matched word
            if last_matched_word_index_in_window != -1:
                transcribed_word_cursor = start_pos + last_matched_word_index_in_window + 1
        else:
            # If no alignment was found in the search window, something is wrong.
            # For now, we'll just advance the cursor by the number of words in the segment
            # as a fallback, but a more robust solution might be needed.
            print(f"Warning: No alignment found for segment ID {segment['id']}. This may indicate a significant mismatch.")
            # Fallback: advance the cursor by an estimated amount
            transcribed_word_cursor += len(segment_words)


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

def main(model, audio_path, text_path, output_dir):
    """
    Main function to run the synchronization process.
    """
    process_audiobook(model, audio_path, text_path, output_dir)

if __name__ == "__main__":
    test_data_dir = Path("c:/Code/Lumi/data")
    output_dir = Path("c:/Code/Lumi/sync_service/output")

    if not test_data_dir.exists():
        print(f"Error: Directory not found at '{test_data_dir}'")
    else:
        audio_file = next(test_data_dir.glob("*.mp3"), None)

        if audio_file:
            text_file = audio_file.with_suffix(".txt")

            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "int8_float16" if device == "cpu" else "int8_float16"
            print("Loading Whisper model...")
            model = WhisperModel("large-v2", device=device, compute_type=compute_type)

            if not text_file.exists():
                print(f"Text file not found for {audio_file.name}. Generating text from audio...")
                segments, _ = model.transcribe(str(audio_file))
                
                full_text = "".join(segment.text for segment in segments)
                
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"Generated text file: {text_file}")

            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            main(model, str(audio_file), str(text_file), str(output_dir))
        else:
            print("No .mp3 file found in the test_data directory.")
