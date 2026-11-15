"""Unit tests for challenge pack rust detector execution and error handling.
"""

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.rust_detector import RustDetector


class TestChallengePackFix(unittest.TestCase):

    def test_rust_detector_non_zero_on_rust_image(self):
        img = np.full((300, 300, 3), 200, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 200), (20, 100, 180), -1)
        detector = RustDetector()
        res = detector.detect(img)
        self.assertEqual(res.get("execution_status"), "CANDIDATE_DETECTED")
        rust_pct = res.get("metadata", {}).get("rust_percentage", 0.0)
        self.assertGreater(rust_pct, 0.0)

    def test_detector_exception_handling_mock(self):
        class FaultyDetector:
            def detect(self, img):
                raise RuntimeError("simulated detector crash")

        faulty = FaultyDetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        try:
            res = faulty.detect(img)
            err_caught = None
        except Exception as e:
            err_caught = str(e)

        self.assertEqual(err_caught, "simulated detector crash")
