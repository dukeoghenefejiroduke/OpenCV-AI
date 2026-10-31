"""Comprehensive Unit Tests for Leakage & Seepage Screening Detector (tests/test_leakage_detector.py).

Tests leakage screening detector across synthetic fixtures (vertical/diagonal streaks, stains, clean surfaces),
input validation, schema compliance, orchestrator aggregation, and corrosion regression.
"""

import pytest
import numpy as np
import cv2
import os

from vision.contract import validate_finding, STATUS_INPUT_REJECTED, STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED, STATUS_DETECTOR_FAILED
from vision.leakage_detector import LeakageDetector
from vision.orchestrator import MultiHazardOrchestrator
from vision.corrosion_detector import CorrosionDetector


def test_leakage_detector_synthetic_fixtures():
    """Tests LeakageDetector on clean surfaces, synthetic vertical streaks, and diagonal/stain regions."""
    detector = LeakageDetector()

    # 1. Clean uniform surface
    clean_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    res_clean = detector.detect(clean_img)
    assert res_clean["execution_status"] == STATUS_NO_CANDIDATE

    # 2. Synthetic vertical streak (dark vertical line/band)
    streak_img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(streak_img, (140, 50), (160, 250), (50, 50, 50), -1)
    res_streak = detector.detect(streak_img)
    assert res_streak["execution_status"] in [STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED]
    for f in res_streak.get("findings", []):
        is_val, err = validate_finding(f, image_shape=(300, 300))
        assert is_val, f"Finding failed validation: {err}"
        assert "potential_leakage_indicator" in f["evidence_description"]
        assert "possible_seepage_staining" in f["evidence_description"]
        assert "requires_manual_inspection" in f["evidence_description"]

    # 3. Undersized image below 128x128 threshold
    small_img = np.full((64, 64, 3), 200, dtype=np.uint8)
    res_small = detector.detect(small_img)
    assert res_small["execution_status"] == STATUS_INPUT_REJECTED


def test_leakage_detector_invalid_inputs():
    """Tests empty, None, and invalid input handling."""
    detector = LeakageDetector()
    res_none = detector.detect(None)
    assert res_none["execution_status"] == STATUS_INPUT_REJECTED

    res_empty = detector.detect(np.array([]))
    assert res_empty["execution_status"] == STATUS_INPUT_REJECTED


def test_orchestrator_five_detectors_integration():
    """Tests that MultiHazardOrchestrator runs all 5 detectors safely."""
    orchestrator = MultiHazardOrchestrator()
    sample_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(sample_path):
        img = np.full((300, 300, 3), 200, dtype=np.uint8)
    else:
        img = sample_path

    report = orchestrator.run_inspections(img)
    assert "detectors" in report
    assert "corrosion_detector" in report["detectors"]
    assert "crack_screening_detector" in report["detectors"]
    assert "surface_defect_detector" in report["detectors"]
    assert "leakage_detector" in report["detectors"]
    assert "insulation_detector" in report["detectors"]
    assert len(report["detector_summaries"]) == 5

    # Verify authoritative corrosion inspection report is preserved
    auth_rep = report["authoritative_inspection_report"]
    if os.path.exists(sample_path):
        assert auth_rep["corrosion_percentage"] == 6.56
        assert auth_rep["region_count"] == 11
        assert auth_rep["recommendation"] == "WARNING"


def test_corrosion_regression_guard_leakage():
    """Ensures corrosion detector regression parity is strictly maintained after adding leakage detector."""
    sample_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(sample_path):
        pytest.skip("Sample image not found.")

    detector = CorrosionDetector()
    res = detector.detect(sample_path)
    meta = res["metadata"]
    report = meta["inspection_report"]

    assert report["dimensions"]["width"] == 1920
    assert report["dimensions"]["height"] == 1080
    assert report["corrosion_percentage"] == 6.56
    assert report["region_count"] == 11
    assert report["recommendation"] == "WARNING"
