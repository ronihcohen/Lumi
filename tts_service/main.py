#!/usr/bin/env python3
"""Kokoro-82M TTS for audiobook generation."""

import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

import soundfile as sf
from kokoro import KPipeline

warnings.filterwarnings("ignore")

MAX_TOKENS = 200


def split_text(text, max_tokens=MAX_TOKENS):
    """Split text into chunks respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        words = sentence.split()
        if len(current) + len(words) <= max_tokens:
            current += " " + sentence if current else sentence
        else:
            if current:
                chunks.append(current)
            if len(words) <= max_tokens:
                current = sentence
            else:
                current = ""
                for word in words:
                    if len(current.split()) + 1 <= max_tokens:
                        current += " " + word if current else word
                    else:
                        if current:
                            chunks.append(current)
                        current = word

    if current:
        chunks.append(current)

    return chunks


def synthesize_chunks(pipeline, chunks, voice, output_file, speed=1.0):
    """Generate audio per chunk and concatenate."""
    print(f"Total chunks: {len(chunks)}")

    audio_arrays = []

    for chunk in tqdm(chunks, desc="Synthesizing", unit="chunk"):
        generator = pipeline(chunk, voice=voice, speed=speed)
        for _, _, audio in generator:
            audio_arrays.append(audio)

    if audio_arrays:
        final_audio = np.concatenate(audio_arrays)
        sf.write(output_file, final_audio, 24000)
        print(f"✓ Saved to {output_file}")
    else:
        print("✗ No audio generated")


def main():
    parser = argparse.ArgumentParser(description="Kokoro-82M TTS audiobook generation")
    parser.add_argument("--txt-dir", type=str, default="../txt")
    parser.add_argument("--output-dir", type=str, default="../ui/data")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Kokoro-82M with voice: {args.voice}")
    pipeline = KPipeline(lang_code='a')

    for txt_file in txt_dir.glob("*.txt"):
        if "Alice" not in txt_file.name:
            continue

        output_file = output_dir / (txt_file.stem + ".mp3")

        print("\n" + "=" * 40)
        print(f"Processing: {txt_file.name}")
        print("=" * 40)

        text = Path(txt_file).read_text(encoding="utf-8")
        text = re.sub(r'\s+', ' ', text).strip()

        print("Splitting text into chunks...")
        chunks = split_text(text)
        print(f"Created {len(chunks)} chunks")

        synthesize_chunks(pipeline, chunks, args.voice, output_file, args.speed)


if __name__ == "__main__":
    main()
