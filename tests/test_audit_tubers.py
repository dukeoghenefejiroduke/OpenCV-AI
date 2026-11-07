import os
import sys
import zipfile
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2

from tools.audit_tubers import dhash_from_bytes, hamming_hex, FRAME_RE, infer_camera, audit


def _jpg_bytes(img_array):
    ok, buf = cv2.imencode(".jpg", img_array)
    assert ok
    return buf.tobytes()


def test_dhash_identical_images_zero_hamming():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    a = dhash_from_bytes(_jpg_bytes(img))
    b = dhash_from_bytes(_jpg_bytes(img))
    assert a is not None and b is not None
    assert hamming_hex(a, b) == 0


def test_dhash_distinct_images_differ():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.zeros((32, 32, 3), dtype=np.uint8)
    b[:, 16:] = 255
    ha = dhash_from_bytes(_jpg_bytes(a))
    hb = dhash_from_bytes(_jpg_bytes(b))
    assert ha is not None and hb is not None
    assert hamming_hex(ha, hb) > 0


def test_frame_regex():
    m = FRAME_RE.match("2511201539a_001808.jpg")
    assert m and m.group("prefix") == "2511201539a" and int(m.group("frame")) == 1808


def test_camera_inference():
    assert "1280x720" in infer_camera("WIN_20200101_000001.jpg")
    assert "1920x1080" in infer_camera("2511201539a_001808.jpg")


def test_audit_against_synthetic_zip(tmp_path):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    zb = _jpg_bytes(img)
    zpath = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        # same image in train and test — should register as near-duplicate
        zf.writestr("train/crack/2511201539a_000001.jpg", zb)
        zf.writestr("test/crack/2511201539a_000002.jpg", zb)
        zf.writestr("val/no-crack/2511201551a_000001.jpg", _jpg_bytes(np.zeros((32, 32, 3), dtype=np.uint8)))
    out = tmp_path / "audit"
    rc = audit(str(zpath), str(out), dup_threshold=4)
    assert rc == 0
    import json
    rep = json.load(open(out / "tubers_audit.json", encoding="utf-8"))
    assert rep["evaluation_eligibility"] == "DISABLED"
    assert rep["cross_partition_near_duplicate_count"] >= 1
    assert "2511201539a" in rep["prefixes_spanning_multiple_partitions"]