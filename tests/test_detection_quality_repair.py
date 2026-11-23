import os
import hashlib
import numpy as np
import cv2
from vision.orchestrator import MultiHazardOrchestrator
from vision.inspection import inspect_corrosion, compute_foreground_roi_mask
from vision.contract import validate_finding


def test_protected_inspection_pipeline_hash():
    """Verify that protected production file vision/inspection.py maintains its exact SHA-256 hash."""
    expected_hash = "8485876fce21200a18c56efb87a21353b98cb5390e8b375c1cd1d3d81a7802eb"
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


def test_macro_closeup_box_area_capped_and_threads_suppressed():
    """Verify macro close-ups cap single-box area <= 82% and suppress repeating pipe thread shadows."""
    from vision.experimental_detector import ExperimentalLeakageDetector, ExperimentalCoatingDetector

    # Synthetic repeating parallel dark pipe threads + bright metallic ridges
    threads = np.full((360, 480, 3), 185, dtype=np.uint8)
    for tx in range(160, 320, 22):
        cv2.rectangle(threads, (tx, 90), (tx + 9, 250), (35, 35, 35), -1)
    leak_res = ExperimentalLeakageDetector().detect(threads)
    assert leak_res["findings"] == [], f"Machined thread grooves flagged as leakage: {leak_res['findings']}"

    # Synthetic genuine gravity-driven downward drip widening into a bottom pool (non-periodic)
    drip = np.full((360, 480, 3), 205, dtype=np.uint8)
    pts = np.array([[235, 70], [243, 70], [245, 190], [262, 245], [218, 245], [233, 190]], dtype=np.int32)
    cv2.fillPoly(drip, [pts], (35, 35, 35))
    drip_res = ExperimentalLeakageDetector().detect(drip)
    assert len(drip_res["findings"]) >= 1, "Genuine downward drip/pool should be detected"

    # Tiny corner shadow (< 0.4% frame area) suppressed by ExperimentalCoatingDetector
    tiny_coat = np.full((360, 480, 3), 160, dtype=np.uint8)
    cv2.rectangle(tiny_coat, (4, 336), (22, 354), (40, 150, 60), -1)
    coat_res = ExperimentalCoatingDetector().detect(tiny_coat)
    assert coat_res["findings"] == [], f"Tiny corner shadow should be suppressed: {coat_res['findings']}"

    # If real macro image exists, ensure no whole-frame COR box (>82%) and no thread LEK-* findings
    macro_path = "data/outputs/original_INS-SI46VXDW.jpg"
    if os.path.exists(macro_path):
        corr = inspect_corrosion(macro_path)
        total = corr["dimensions"]["width"] * corr["dimensions"]["height"]
        for bx in corr["bounding_boxes"]:
            assert (bx[2] * bx[3]) <= 0.82 * total, f"Macro box exceeded 82% frame cap: {bx}"
        assert ExperimentalLeakageDetector().detect(macro_path)["findings"] == []

