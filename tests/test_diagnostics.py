"""Unit test suite for CementSight Diagnostic Visual Processing Explorer (tests/test_diagnostics.py).

Tests diagnostic stage generation, non-interference with production inspection results,
path security, and diagnostic API responses.
"""

import unittest
import os
import sys
import json
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.inspection import inspect_corrosion
from vision.diagnostics import generate_diagnostic_stages
from app.validation import validate_image_input


class TestDiagnostics(unittest.TestCase):

    def test_1_diagnostic_stages_generation(self):
        img_path = "data/pipe_corrosion.jpg"
        self.assertTrue(os.path.exists(img_path), "Test image must exist")

        stages = generate_diagnostic_stages(img_path, output_dir="data/outputs/diagnostics_test")
        self.assertIsInstance(stages, list)
        self.assertEqual(len(stages), 7)

        expected_ids = ["original", "blurred", "raw_hsv_mask", "roi_mask", "restricted_mask", "post_morph_mask", "annotated"]
        for i, stage in enumerate(stages):
            self.assertEqual(stage["stage_id"], expected_ids[i])
            self.assertTrue(os.path.exists(stage["image_url"]), f"Diagnostic image for {stage['stage_id']} must exist on disk")

    def test_2_diagnostics_do_not_alter_inspection_result(self):
        img_path = "data/pipe_corrosion.jpg"
        img = cv2.imread(img_path)

        # Run standard inspection without diagnostics
        report_standard = inspect_corrosion(img, use_roi=True)

        # Run diagnostics (which also calls inspect_corrosion internally for stage 7)
        stages = generate_diagnostic_stages(img, output_dir="data/outputs/diagnostics_test")

        # Run standard inspection again
        report_after = inspect_corrosion(img, use_roi=True)

        self.assertEqual(report_standard["corrosion_percentage"], report_after["corrosion_percentage"])
        self.assertEqual(report_standard["region_count"], report_after["region_count"])
        self.assertEqual(report_standard["recommendation"], report_after["recommendation"])

    def test_3_invalid_image_rejected_before_diagnostics(self):
        is_valid, decoded, err = validate_image_input("non_existent.jpg")
        self.assertFalse(is_valid)
        with self.assertRaises(FileNotFoundError):
            generate_diagnostic_stages("non_existent.jpg")


if __name__ == "__main__":
    unittest.main()
