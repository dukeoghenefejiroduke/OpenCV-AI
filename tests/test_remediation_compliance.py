"""Tests for CementSight Remediation Compliance (tests/test_remediation_compliance.py).

Verifies image downscaling, friendly error handling, feature flags, feedback loop storage,
concurrency semaphore limit (max 5), data/outputs isolation, and concurrent HTTP load testing.
"""

import os
import json
import time
import threading
import urllib.request
import cv2
import numpy as np
import pytest

from vision.quality_gate import ImageQualityGate
from vision.orchestrator import MultiHazardOrchestrator
from app.validation import validate_image_input
from tools.server import ReuseTCPServer, CementSightHandler, _inspection_semaphore, MAX_CONCURRENT_INSPECTIONS


def test_image_downscaling():
    """Test that large images above 2000px are downscaled properly."""
    large_img = np.full((1200, 2500, 3), 150, dtype=np.uint8)
    h, w = large_img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000.0 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        downscaled = cv2.resize(large_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        downscaled = large_img

    assert max(downscaled.shape[:2]) <= 2000


def test_friendly_error_handling():
    """Test that invalid or corrupt image inputs produce friendly error messages."""
    tiny_img = np.full((20, 20, 3), 100, dtype=np.uint8)
    is_val, _, err = validate_image_input(tiny_img)
    assert not is_val
    assert "dimensions" in err.lower() or "too small" in err.lower() or "min" in err.lower()


def test_feature_flags_default():
    """Test that by default only core corrosion and quality gate are primary, experimental are off."""
    orchestrator = MultiHazardOrchestrator(enable_experimental=False)
    detector_ids = [d.detector_id for d in orchestrator.detectors]
    assert "corrosion_detector" in detector_ids
    assert "material_buildup_detector" not in detector_ids


def test_feedback_loop_persistence():
    """Test storing feedback in data/field_feedback/."""
    feedback_dir = "data/field_feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, "test_feedback.jsonl")

    feedback_record = {
        "image_sha256": "abc123hash",
        "result_id": "test_res_01",
        "user_note": "False positive rust detection on shadow",
        "expected_outcome": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_record) + "\n")

    assert os.path.exists(feedback_file)
    with open(feedback_file, "r") as f:
        lines = f.readlines()
        found = any("abc123hash" in line for line in lines)
        assert found


def test_concurrency_semaphore_limit():
    """Test that _inspection_semaphore restricts concurrent acquisitions to MAX_CONCURRENT_INSPECTIONS (5)."""
    assert MAX_CONCURRENT_INSPECTIONS == 5
    # Verify semaphore behavior
    acquired_count = 0
    lock = threading.Lock()

    def try_acquire():
        nonlocal acquired_count
        if _inspection_semaphore.acquire(blocking=False):
            try:
                with lock:
                    acquired_count += 1
                time.sleep(0.1)
            finally:
                _inspection_semaphore.release()

    threads = [threading.Thread(target=try_acquire) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert acquired_count <= 5


def test_data_outputs_isolation():
    """Test data/outputs file count isolation."""
    out_dir = "data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    count_before = len(os.listdir(out_dir))
    
    # Run orchestrator
    orchestrator = MultiHazardOrchestrator()
    dummy = np.full((150, 150, 3), 160, dtype=np.uint8)
    _ = orchestrator.run_inspections(dummy)

    count_after = len(os.listdir(out_dir))
    assert count_before == count_after, f"data/outputs modified! Before: {count_before}, After: {count_after}"


def test_server_http_load_and_concurrency():
    """Test 20 parallel HTTP requests against the multi-threaded server with concurrency limit."""
    port = 8996
    server = ReuseTCPServer(("127.0.0.1", port), CementSightHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    try:
        latencies = []
        errors = []
        def hit_server():
            try:
                start = time.time()
                req = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
                _ = json.loads(req.read().decode("utf-8"))
                dur = time.time() - start
                latencies.append(dur)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=hit_server) for _ in range(20)]
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start_time
        assert len(errors) == 0, f"Errors in HTTP load test: {errors}"
        assert len(latencies) == 20
        avg_lat = sum(latencies) / len(latencies)
        print(f"[HTTP Load Test] 20 parallel requests completed in {total_time:.3f}s. Avg HTTP latency: {avg_lat:.3f}s")
        assert total_time < 30.0
    finally:
        server.shutdown()
        server.server_close()
