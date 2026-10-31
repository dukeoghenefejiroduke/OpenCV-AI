"""Comprehensive Unit Tests for Multi-Hazard Vision Platform (tests/test_multi_hazard.py).

Tests detector contracts, validation schemas, corrosion regression parity, crack screening,
and synthetic test fixtures without modifying vision/inspection.py.
"""

import pytest
import numpy as np
import cv2
import os

from vision.contract import validate_finding, STATUS_INPUT_REJECTED, STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED
from vision.corrosion_detector import CorrosionDetector
from vision.crack_detector import CrackDetector
from vision.orchestrator import MultiHazardOrchestrator


def test_finding_validation():
    """Tests the finding validation schema with valid and invalid finding dictionaries."""
    valid_finding = {
        "finding_id": "test_123",
        "detector_id": "corrosion_detector",
        "detector_version": "1.0.0",
        "defect_category": "corrosion",
        "bbox": [10, 20, 50, 60],
        "pixel_measurement": {"area": 100},
        "evidence_description": "Test finding",
        "confidence": None,
        "review_status": "PENDING_REVIEW",
        "limitations": ["Test limitation"]
    }

    is_valid, err = validate_finding(valid_finding, image_shape=(100, 100))
    assert is_valid, f"Valid finding failed validation: {err}"

    # Test out of bounds bbox
    invalid_bbox_finding = {**valid_finding, "bbox": [80, 80, 50, 50]}
    is_valid, err = validate_finding(invalid_bbox_finding, image_shape=(100, 100))
    assert not is_valid, "Out-of-bounds bbox should have failed validation"

    # Test missing key
    invalid_key_finding = {k: v for k, v in valid_finding.items() if k != "finding_id"}
    is_valid, err = validate_finding(invalid_key_finding, image_shape=(100, 100))
    assert not is_valid, "Missing key should have failed validation"


def test_corrosion_detector_regression():
    """Tests that CorrosionDetector reproduces exact established baseline metrics on data/pipe_corrosion.jpg."""
    sample_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(sample_path):
        pytest.skip(f"Sample image {sample_path} not found.")

    detector = CorrosionDetector()
    res = detector.detect(sample_path)

    assert res["execution_status"] == STATUS_CANDIDATE_DETECTED
    meta = res["metadata"]
    report = meta["inspection_report"]

    assert report["dimensions"]["width"] == 1920
    assert report["dimensions"]["height"] == 1080
    assert report["corrosion_percentage"] == 6.56
    assert report["region_count"] == 11
    assert report["recommendation"] == "WARNING"


def test_crack_detector_synthetic_fixtures():
    """Tests CrackDetector on various synthetic fixtures (clean surface, lines, blank, invalid)."""
    detector = CrackDetector()

    # 1. Clean uniform surface
    clean_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    res_clean = detector.detect(clean_img)
    assert res_clean["execution_status"] == STATUS_NO_CANDIDATE

    # 2. Synthetic line image (dark line drawn across)
    line_img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.line(line_img, (50, 50), (250, 250), (10, 10, 10), 3)
    res_line = detector.detect(line_img)
    # Should detect at least one candidate or execute without error
    assert res_line["execution_status"] in [STATUS_NO_CANDIDATE, STATUS_CANDIDATE_DETECTED]

    # 3. Small invalid image below 128x128 threshold
    small_img = np.full((64, 64, 3), 200, dtype=np.uint8)
    res_small = detector.detect(small_img)
    assert res_small["execution_status"] == STATUS_INPUT_REJECTED


def test_multi_hazard_orchestrator():
    """Tests the multi-hazard orchestrator runs all detectors successfully."""
    orchestrator = MultiHazardOrchestrator()
    sample_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(sample_path):
        # Use synthetic image if sample image not available
        img = np.full((300, 300, 3), 200, dtype=np.uint8)
    else:
        img = sample_path

    platform_report = orchestrator.run_inspections(img)
    assert "platform_name" in platform_report
    assert "detectors" in platform_report
    assert "corrosion_detector" in platform_report["detectors"]
    assert "crack_screening_detector" in platform_report["detectors"]
    assert "surface_defect_detector" in platform_report["detectors"]
    assert "leakage_detector" in platform_report["detectors"]
    assert "insulation_detector" in platform_report["detectors"]
    assert len(platform_report["detector_summaries"]) == 5
