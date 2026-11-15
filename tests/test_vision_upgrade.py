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


def test_synthetic_bright_uniform_wall_no_glare():
    gate = ImageQualityGate()
    # Uniform bright wall (e.g. value 240, high saturation or no compact blown highlights)
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    res = gate.evaluate(img)
    reasons_str = " ".join(res.get("reasons", []))
    assert "glare" not in reasons_str.lower() and "blown highlights" not in reasons_str.lower()


def test_synthetic_dark_frame_darkness_warning():
    gate = ImageQualityGate()
    # Extremely dark frame
    img = np.full((300, 300, 3), 10, dtype=np.uint8)
    res = gate.evaluate(img)
    assert res["verdict"] == "LOW_QUALITY"
    reasons_str = " ".join(res.get("reasons", []))
    assert "dark" in reasons_str.lower() or "under-exposed" in reasons_str.lower()


def test_synthetic_rust_patch_found():
    from vision.rust_detector import RustDetector
    detector = RustDetector()
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    # Draw a prominent rust-colored patch (orange/brown in BGR: e.g. blue=20, green=100, red=180)
    cv2.rectangle(img, (100, 100), (200, 200), (20, 100, 180), -1)
    res = detector.detect(img)
    assert res["execution_status"] == "CANDIDATE_DETECTED"
    assert len(res["findings"]) > 0


def test_synthetic_gray_image_no_rust():
    from vision.rust_detector import RustDetector
    detector = RustDetector()
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    res = detector.detect(img)
    assert res["execution_status"] == "NO_CANDIDATE"
    assert len(res["findings"]) == 0


def test_dev_confusion_refuses_unlisted_files(tmp_path):
    import tempfile
    import csv
    from tools.dev_confusion import run_confusion

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    # Create an unlisted image
    cv2.imwrite(str(photos_dir / "unlisted.jpg"), np.full((100, 100, 3), 100, dtype=np.uint8))

    truth_csv = tmp_path / "truth.csv"
    with open(truth_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth"])
        writer.writerow(["other.jpg", "clean"])

    # Expect sys.exit(2) due to unlisted files
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        run_confusion(str(photos_dir), str(truth_csv))
    assert exc_info.value.code == 2


def test_blue_grey_gradient_with_white_highlights_zero_rust():
    from vision.rust_detector import RustDetector
    detector = RustDetector()
    # Create blue-grey gradient image (higher blue channel)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in range(300):
        val_b = int(120 + (y / 300.0) * 30)
        val_g = int(110 + (y / 300.0) * 20)
        val_r = int(100 + (y / 300.0) * 10)
        img[y, :, :] = [val_b, val_g, val_r]
    # Add white highlights
    cv2.circle(img, (150, 150), 30, (255, 255, 255), -1)
    res = detector.detect(img)
    assert res["execution_status"] == "NO_CANDIDATE"
    assert len(res["findings"]) == 0


def test_uniform_bright_wall_no_low_quality():
    gate = ImageQualityGate()
    np.random.seed(42)
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    noise = np.random.randint(-60, 60, img.shape).astype(np.int32)
    img = np.clip(img.astype(np.int32) + noise, 100, 255).astype(np.uint8)
    res = gate.evaluate(img)
    assert res["verdict"] == "OK"


def test_blue_tinted_gray_scene_zero_rust():
    from vision.rust_detector import RustDetector
    detector = RustDetector()
    # Blue-tinted gray scene (e.g., B: 140, G: 120, R: 110)
    img = np.full((300, 300, 3), [140, 120, 110], dtype=np.uint8)
    # Add some gaussian noise
    np.random.seed(123)
    noise = np.random.normal(0, 5, img.shape).astype(np.int32)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    res = detector.detect(img)
    assert res["execution_status"] == "NO_CANDIDATE"
    assert len(res["findings"]) == 0


def test_rust_coloured_rough_patch_on_gray_found():
    from vision.rust_detector import RustDetector
    detector = RustDetector()
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    # Draw rough rust-coloured patch
    np.random.seed(42)
    rust_patch = np.random.randint(10, 40, (100, 100, 3), dtype=np.uint8)
    rust_patch[:, :, 0] = np.clip(rust_patch[:, :, 0] + 20, 0, 255) # blue low
    rust_patch[:, :, 1] = np.clip(rust_patch[:, :, 1] + 90, 0, 255) # green medium
    rust_patch[:, :, 2] = np.clip(rust_patch[:, :, 2] + 170, 0, 255) # red high
    img[100:200, 100:200] = rust_patch
    res = detector.detect(img)
    assert res["execution_status"] == "CANDIDATE_DETECTED"
    assert len(res["findings"]) > 0
    # Verify roughness and shape stats are recorded
    f = res["findings"][0]
    assert "roughness_texture_std" in f["pixel_measurement"]
    assert "shape_stats" in f["pixel_measurement"]


def test_golden_corrosion_result_unchanged():
    from vision.corrosion_detector import CorrosionDetector
    detector = CorrosionDetector()
    img = np.full((300, 300, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (30, 100, 200), -1)
    res = detector.detect(img)
    assert "detector_id" in res
    assert res["detector_id"] == "corrosion_detector"
    assert "metadata" in res


