import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import os
import sys

# Ensure sync_service is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import get_audio_duration, load_audio_chunk, needleman_wunsch, save_sync_outputs

class TestSyncServiceHelper(unittest.TestCase):

    @patch("subprocess.check_output")
    def test_get_audio_duration(self, mock_sub):
        mock_sub.return_value = b"123.45\n"
        dur = get_audio_duration("dummy.mp3")
        self.assertEqual(dur, 123.45)
        
    @patch("subprocess.Popen")
    def test_load_audio_chunk(self, mock_popen):
        # Create a mock process with stdout
        process_mock = MagicMock()
        
        # Simulate 1 second of 16kHz audio (16000 samples)
        # s16le = 2 bytes per sample
        fake_audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()
        
        process_mock.communicate.return_value = (fake_audio_bytes, None)
        mock_popen.return_value = process_mock
        
        audio = load_audio_chunk("dummy.mp3", 0, 1)
        
        self.assertEqual(len(audio), 16000)
        self.assertEqual(audio.dtype, np.float32)

    def test_needleman_wunsch(self):
        seq1 = ["The", "quick", "brown"]
        # Exact match
        seq2 = [
            {"word": "The", "start": 0, "end": 1},
            {"word": "quick", "start": 1, "end": 2},
            {"word": "brown", "start": 2, "end": 3}
        ]
        
        aligned = needleman_wunsch(seq1, seq2)
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned[0][0], "The")
        self.assertEqual(aligned[0][1]["word"], "The")
        
        # Mismatch
        seq1_diff = ["The", "fast", "brown"]
        aligned_diff = needleman_wunsch(seq1_diff, seq2)
        
        # Inspect alignment for "fast" vs "quick"
        # Since gap penalty is -1 and mismatch is -1, it might align them as mismatch or gap
        found_fast = False
        for s1, s2 in aligned_diff:
            if s1 == "fast":
                found_fast = True
        self.assertTrue(found_fast)

    def test_save_sync_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            save_sync_outputs(
                temp_path, 
                "test", 
                {"p1": {"start": 0, "end": 1}}, 
                [], 
                [{"id": "p1", "text": "foo"}]
            )
            
            self.assertTrue((temp_path / "test_sync_map.json").exists())
            self.assertTrue((temp_path / "test_transcribed_words.json").exists())
            self.assertTrue((temp_path / "test_segments.json").exists())

if __name__ == "__main__":
    unittest.main()
