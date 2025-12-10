import argparse
import torch
import re
from pathlib import Path
from TTS.api import TTS
from pydub import AudioSegment
import tempfile
from tqdm import tqdm

MAX_CHARS = 200   # safe for XTTS


def split_text(text, max_chars=MAX_CHARS):
    """Split text manually into safe ~200 char chunks."""
    chunks = []
    current = ""

    for word in text.split():
        if len(current) + len(word) + 1 <= max_chars:
            current += " " + word if current else word
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def synthesize_chunks(tts, chunks, speaker_wav, language, output_file):
    """Generate audio per chunk and join into one MP3 using tqdm."""
    print(f"Total chunks: {len(chunks)}")

    final_audio = AudioSegment.silent(duration=0)

    # tqdm progress bar
    for chunk in tqdm(chunks, desc="Synthesizing chunks", unit="chunk"):
        # temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # synthesize one chunk
        tts.tts_to_file(
            text=chunk,
            file_path=tmp_path,
            speaker_wav=speaker_wav,
            language=language,
            enable_text_splitting=False  # we do manual splitting
        )

        audio = AudioSegment.from_wav(tmp_path)
        final_audio += audio

    final_audio.export(output_file, format="mp3")
    print("✔ Final MP3 saved.")


def main():
    parser = argparse.ArgumentParser(description="Batch TTS using XTTS-v2")
    parser.add_argument("--txt-dir", type=str, default="../txt")
    parser.add_argument("--output-dir", type=str, default="../data")
    parser.add_argument("--speaker-wav", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing XTTS-v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    speaker_wav = args.speaker_wav or "speaker_ref.wav"

    for txt_file in txt_dir.glob("*.txt"):
        output_file = output_dir / (txt_file.stem + ".mp3")

        print("\n==============================")
        print(f"Processing: {txt_file.name}")
        print("==============================")

        text = Path(txt_file).read_text(encoding="utf-8")
        text = re.sub(r'\s+', ' ', text).strip()

        print("Splitting text into chunks…")
        chunks = split_text(text)

        synthesize_chunks(tts, chunks, speaker_wav, args.language, output_file)


if __name__ == "__main__":
    main()
