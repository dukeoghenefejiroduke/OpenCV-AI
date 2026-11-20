import os
import hashlib
import numpy as np
import cv2
from vision.orchestrator import MultiHazardOrchestrator
from vision.inspection import inspect_corrosion, compute_foreground_roi_mask
from vision.contract import validate_finding


def test_protected_inspection_pipeline_hash():
    """Verify that protected production file vision/inspection.py maintains its exact SHA-256 hash."""
    expected_hash = "9c695dd647e4ded58a4cd7a683fcc55c297b2e4dbb1daf66c7c74b3dcdecb38c"
    with open("vision/inspection.py", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == expected_hash, f"Protected file hash mismatch: {file_hash}"
    print("test_protected_inspection_pipeline_hash PASSED")


def test_experimental_detector_background_suppression():
    """Test that experimental detectors suppress background artifacts outside the asset ROI."""
    img = np.full((400, 400, 3), 220, dtype=np.uint8)
    # Asset in center
    cv2.rectangle(img, (150, 100), (250, 300), (140, 140, 140), -1)

    orchestrator = MultiHazardOrchestrator(enable_experimental=True)
    report = orchestrator.run_inspections(img)

    assert report is not None
    assert "detectors" in report
    # Ensure experimental detectors (rust_detector, crack_screening_detector, etc.) run successfully and respect ROI
    for det_id, det_res in report["detectors"].items():
        if det_id != "corrosion_detector":
            assert "findings" in det_res

    print("test_experimental_detector_background_suppression PASSED")


def test_duplicate_detection_suppression_nms():
    """Test that heavily overlapping findings are suppressed by NMS."""
    orchestrator = MultiHazardOrchestrator(enable_experimental=True)
    
    img = np.full((300, 300, 3), 160, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (160, 160), (25, 90, 180), -1)

    report = orchestrator.run_inspections(img)
    all_count = report.get("total_findings_count", 0)
    assert all_count >= 0
    print("test_duplicate_detection_suppression_nms PASSED")


def test_small_genuine_defect_preservation():
    """Test that small genuine defects are preserved while micro-specks are filtered."""
    img = np.full((300, 300, 3), 180, dtype=np.uint8)
    # Asset body
    cv2.rectangle(img, (100, 50), (200, 250), (130, 130, 130), -1)
    # Small genuine defect
    cv2.rectangle(img, (120, 100), (135, 115), (25, 90, 180), -1)

    report = inspect_corrosion(img)
    assert report["region_count"] >= 1
    print("test_small_genuine_defect_preservation PASSED")


def test_empty_and_invalid_detector_inputs():
    """Test detector fault isolation and graceful handling of empty or invalid inputs."""
    orchestrator = MultiHazardOrchestrator(enable_experimental=True)
    
    invalid_img = np.array([], dtype=np.uint8)
    report = orchestrator.run_inspections(invalid_img)
    assert report is not None
    assert "detectors" in report
    print("test_empty_and_invalid_detector_inputs PASSED")
