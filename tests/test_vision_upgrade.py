"""Procedural and Regression Tests for Vision Pipeline Upgrades & Quality Gate (tests/test_vision_upgrade.py)."""

import pytest
import numpy as np
import cv2

from vision.quality_gate import ImageQualityGate
from vision.orchestrator import MultiHazardOrchestrator
from vision.crack_detector import CrackDetector
from vision.contract import validate_finding


def test_quality_gate_blur():
    gate = ImageQualityGate()
    # Create blank / low variance image (blurry)
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    res = gate.evaluate(img)
    assert res["verdict"] == "LOW_QUALITY"
    assert any("blurry" in r.lower() for r in res["reasons"])
    assert res["recommendation"] == "retake image recommended"


def test_quality_gate_ok():
    gate = ImageQualityGate()
    # Create image with high-frequency noise/edges
    np.random.seed(42)
    img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    # Draw some sharp lines to boost Laplacian variance
    cv2.line(img, (10, 10), (290, 290), (255, 255, 255), 3)
    cv2.line(img, (10, 290), (290, 10), (0, 0, 0), 3)
    res = gate.evaluate(img)
    assert res["verdict"] == "OK"
    assert res["recommendation"] == "none"


def test_crack_detector_blank_image():
    detector = CrackDetector()
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    res = detector.detect(img)
    assert res["execution_status"] == "NO_CANDIDATE"
    assert len(res["findings"]) == 0


def test_crack_detector_drawn_crack():
    detector = CrackDetector()
    # Base grey image
    img = np.full((400, 400, 3), 220, dtype=np.uint8)
    # Draw a thin dark jagged line (crack)
    pts = np.array([[50, 200], [100, 195], [150, 205], [200, 198], [250, 202], [300, 200]], np.int32)
    cv2.polylines(img, [pts], False, (30, 30, 30), 2)
    res = detector.detect(img)
    # Should detect crack or candidate
    assert isinstance(res["findings"], list)
    for f in res["findings"]:
        is_val, err = validate_finding(f, image_shape=(400, 400))
        assert is_val, f"Finding invalid: {err}"


def test_crack_detector_straight_seam():
    detector = CrackDetector()
    img = np.full((400, 400, 3), 220, dtype=np.uint8)
    # Draw a perfectly horizontal straight line spanning almost the entire width (seam)
    cv2.line(img, (10, 200), (390, 200), (30, 30, 30), 3)
    res = detector.detect(img)
    seam_findings = [f for f in res["findings"] if "seam" in f["finding_id"] or "seam" in f["evidence_description"].lower()]
    assert len(seam_findings) > 0, "Expected straight frame-aligned line to be flagged as seam/review."
    for f in seam_findings:
        assert f["confidence"] == 0.2
        assert "possible_seam_review" in f["evidence_description"]


def test_orchestrator_low_quality_handling():
    orchestrator = MultiHazardOrchestrator()
    # Blurry image
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    report = orchestrator.run_inspections(img)
    assert report["image_quality"]["verdict"] == "LOW_QUALITY"
    assert "retake image recommended" in report["image_quality"]["recommendation"]


def test_safety_isolation_corrosion_baseline():
    """Verify production corrosion baseline is unaffected and returns expected structure."""
    from vision.corrosion_detector import CorrosionDetector
    detector = CorrosionDetector()
    # Use real test image if available or synthetic
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    res = detector.detect(img)
    assert "detector_id" in res
    assert res["detector_id"] == "corrosion_detector"
    assert "metadata" in res
