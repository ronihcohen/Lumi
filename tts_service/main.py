import argparse
import torch
import re
from pathlib import Path
from tqdm import tqdm
from TTS.api import TTS

def main():
    """
    Main function to parse arguments, initialize the XTTS model, and process 
    text files in a batch, applying necessary text cleaning for smooth audio flow.
    """
    parser = argparse.ArgumentParser(description="Batch TTS using XTTS-v2")
    parser.add_argument("--txt-dir", type=str, default="../txt", help="Directory containing .txt files")
    parser.add_argument("--output-dir", type=str, default="../data", help="Directory to save .mp3 files")
    parser.add_argument("--speaker-wav", type=str, default=None, help="Path to a reference audio file for voice cloning (required for XTTS)")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    
    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)
    
    if not txt_dir.exists():
        print(f"Error: Text directory not found: {txt_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing XTTS-v2 on {device}...")
    
    try:
        # Load the model and move it to the determined device
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Speaker Reference Check ---
    speaker_wav = args.speaker_wav
    if not speaker_wav:
        default_ref = Path("speaker_ref.wav")
        if default_ref.exists():
            speaker_wav = str(default_ref)
        else:
            print("\nWARNING: XTTS-v2 requires a speaker reference audio file (for voice cloning).")
            print("Please provide one using --speaker-wav or place 'speaker_ref.wav' in this directory.")
            return

    txt_files = list(txt_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files to process.")

    # --- Batch Processing with Progress Bar ---
    # tqdm wraps the list of files to display the progress bar
    for txt_file in tqdm(txt_files, desc="Synthesizing", unit="file"):
        output_file = output_dir / (txt_file.stem + ".mp3")
        
        if output_file.exists():
            tqdm.write(f"Skipping {txt_file.name} (Output exists)")
            continue
            
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text_content = f.read()

            if not text_content.strip():
                tqdm.write(f"Empty file: {txt_file.name}, skipping.")
                continue

            # --- Text Flow Fix ---
            # 1. Replace line breaks (\n, \r) with a single space to prevent harsh pauses.
            clean_text = text_content.replace("\n", " ").replace("\r", " ")
            
            # 2. Substitute multiple consecutive spaces with a single space.
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            # ---------------------

            # XTTS synthesis with optimized parameters for natural flow
            tts.tts_to_file(
                text=clean_text,
                file_path=str(output_file),
                speaker_wav=speaker_wav,
                language=args.language,
                enable_text_splitting=True
            )
            
            tqdm.write(f"✅ Saved: {output_file.name}")
            
        except Exception as e:
            tqdm.write(f"❌ Failed to process {txt_file.name}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()