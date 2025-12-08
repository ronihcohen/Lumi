import os
import argparse
from pathlib import Path
from TTS.api import TTS
import torch

def generate_audio(txt_path, output_path, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"):
    """
    Generates audio from a text file using XTTS-v2.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading model: {model_name} on {device}...")
    tts = TTS(model_name=model_name).to(device)

    print(f"Reading text from {txt_path}...")
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    pass

def main():
    parser = argparse.ArgumentParser(description="Batch TTS using XTTS-v2")
    parser.add_argument("--txt-dir", type=str, default="../txt", help="Directory containing .txt files")
    parser.add_argument("--output-dir", type=str, default="../data", help="Directory to save .mp3 files")
    parser.add_argument("--speaker-wav", type=str, default=None, help="Path to a reference audio file for voice cloning (required for XTTS)")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    
    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)
    
    if not txt_dir.exists():
        print(f"Text directory not found: {txt_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing XTTS-v2 on {device}...")
    
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check for speaker reference
    speaker_wav = args.speaker_wav
    if not speaker_wav:
        # Check if there is a default one in current dir
        default_ref = Path("speaker_ref.wav")
        if default_ref.exists():
            speaker_wav = str(default_ref)
        else:
            print("WARNING: XTTS-v2 requires a speaker reference audio file (for voice cloning).")
            print("Please provide one using --speaker-wav or place 'speaker_ref.wav' in this directory.")
            # We can't proceed easily without a speaker for XTTS usually unless we use speaker_idx which might not be available or stable.
            # I'll exit if no speaker ref is found to avoid waste.
            return

    txt_files = list(txt_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files.")

    for txt_file in txt_files:
        output_file = output_dir / (txt_file.stem + ".mp3")
        
        if output_file.exists():
            print(f"Skipping {txt_file.name} (Output exists)")
            continue
            
        print(f"Processing {txt_file.name}...")
        
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text_content = f.read()

            if not text_content.strip():
                print("Empty file, skipping.")
                continue

            # XTTS synthesis
            # Note: tts_to_file is the high level API method
            tts.tts_to_file(
                text=text_content, 
                file_path=str(output_file),
                speaker_wav=speaker_wav,
                language=args.language
            )
            print(f"Saved to {output_file}")
            
        except Exception as e:
            print(f"Failed to process {txt_file.name}: {e}")

if __name__ == "__main__":
    main()
