"""Unit test suite for Challenge Pack evaluation tool and separation of vision detectors from challenge pack data."""

import unittest
import os
import sys
import subprocess
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestChallengePack(unittest.TestCase):

    def test_vision_code_does_not_read_challenge_pack(self):
        """Verifies that no tuning or detector code under vision/ reads data/challenge_pack."""
        vision_files = glob.glob("vision/**/*.py", recursive=True)
        for vf in vision_files:
            with open(vf, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertNotIn("challenge_pack", content, f"Vision file {vf} references challenge_pack!")

    def test_challenge_pack_refuses_missing_truth_csv(self):
        """Verifies challenge_pack tool refuses or exits if truth CSV is missing."""
        # Temporarily rename truth.csv if present
        truth_path = "data/challenge_pack/truth.csv"
        temp_path = "data/challenge_pack/truth.csv.bak"
        existed = False
        if os.path.exists(truth_path):
            os.rename(truth_path, temp_path)
            existed = True
        try:
            res = subprocess.run([sys.executable, "tools/challenge_pack.py"], capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("not found", res.stderr.lower() + res.stdout.lower())
        finally:
            if existed and os.path.exists(temp_path):
                os.rename(temp_path, truth_path)
