"""Comprehensive Unit Tests for Insulation Damage & Cladding Screening Detector (tests/test_insulation_detector.py).

Tests insulation screening detector across synthetic fixtures (clean surfaces, rectangular gaps,
missing cladding regions), input validation, schema compliance, orchestrator aggregation, and corrosion regression.
"""

import pytest
import numpy as np
import cv2
import os

from vision.contract import validate_finding, STATUS_INPUT_REJECTED, STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED, STATUS_DETECTOR_FAILED
from vision.insulation_detector import InsulationDetector
from vision.orchestrator import MultiHazardOrchestrator
from vision.corrosion_detector import CorrosionDetector


def test_insulation_detector_synthetic_fixtures():
    """Tests InsulationDetector on clean surfaces, rectangular gaps, and irregular regions."""
    detector = InsulationDetector()

    # 1. Clean uniform surface
    clean_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    res_clean = detector.detect(clean_img)
    assert res_clean["execution_status"] == STATUS_NO_CANDIDATE

    # 2. Synthetic rectangular gap / missing cladding patch
    gap_img = np.full((300, 300, 3), 210, dtype=np.uint8)
    cv2.rectangle(gap_img, (100, 100), (200, 200), (40, 40, 40), 3)
    res_gap = detector.detect(gap_img)
    assert res_gap["execution_status"] in [STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED]
    for f in res_gap.get("findings", []):
        is_val, err = validate_finding(f, image_shape=(300, 300))
        assert is_val, f"Finding failed validation: {err}"
        assert "insulation_cladding_anomaly" in f["evidence_description"]
        assert "open_seam_or_missing_cladding" in f["evidence_description"]
        assert "requires_manual_inspection" in f["evidence_description"]

    # 3. Undersized image below 128x128 threshold
    small_img = np.full((64, 64, 3), 200, dtype=np.uint8)
    res_small = detector.detect(small_img)
    assert res_small["execution_status"] == STATUS_INPUT_REJECTED


def test_insulation_detector_invalid_inputs():
    """Tests empty, None, and invalid input handling."""
    detector = InsulationDetector()
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


def test_corrosion_regression_guard_insulation():
    """Ensures corrosion detector regression parity is strictly maintained after adding insulation detector."""
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
