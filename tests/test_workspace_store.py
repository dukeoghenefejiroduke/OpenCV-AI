import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, Finding

# Minimal valid 1x1 JPEG (magic bytes + tiny payload) — sniff passes, bytes preserved.
JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
    "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
    "1c28372f2c31313434342b30363a3d383b3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a"
    "3a3a3a3a3affc0000b080001000101011100ffc4001f00000105010101010101000000"
    "00000000000102030405060708090a0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114328191a1082342b1c11552d1f024"
    "33627282090a161718191a25262728292a3435363738393a434445464748494a535455"
    "565758595a636465666768696a737475767778797a838485868788898a929394959697"
    "98999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5"
    "d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f"
    "00fbfeffd9"
)


def test_image_validation_rejects_non_image(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    with pytest.raises(ValueError):
        s.store_image(b"not an image")


def test_image_validation_rejects_oversize(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    with pytest.raises(ValueError):
        s.store_image(b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024))


def test_image_storage_dedup(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    sha1, ext1 = s.store_image(JPEG_1PX)
    sha2, ext2 = s.store_image(JPEG_1PX)
    assert sha1 == sha2
    assert ext1 == ext2 == "jpg"
    path = s.image_path(sha1, ext1)
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == JPEG_1PX


def test_image_path_rejects_traversal(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    with pytest.raises(ValueError):
        s.image_path("../../etc/passwd", "jpg")
    with pytest.raises(ValueError):
        s.image_path("a" * 64, "exe")


def test_asset_crud(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    a = Asset(asset_id="P-01", location="Plant A", pipe_material="steel", pipe_type="main", criticality="high")
    s.upsert_asset(a)
    got = s.get_asset("P-01")
    assert got is not None and got.criticality == "high"
    assert len(s.list_assets()) == 1


def test_asset_invalid_criticality(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    with pytest.raises(ValueError):
        s.upsert_asset(Asset(asset_id="X", location="y", criticality="weird"))


def test_inspection_and_findings_roundtrip(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    s.upsert_asset(Asset(asset_id="P-01", location="Plant A"))
    sha, ext = s.store_image(JPEG_1PX)
    insp = Inspection(
        inspection_id="INS-1", asset_id="P-01", inspection_date="2026-11-05",
        image_sha256=sha, image_ext=ext,
        detector_results={}, authoritative_report={"corrosion_percentage": 1.0},
    )
    s.create_inspection(insp)
    f = Finding(
        finding_id="F-1", inspection_id="INS-1", detector_id="crack_detector",
        defect_category="crack", bbox=[1, 2, 10, 10], confidence=0.4,
        evidence_description="synthetic", limitations=["heuristic"],
    )
    s.add_findings([f])
    assert len(s.list_findings("INS-1")) == 1


def test_human_review_transition(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    s.upsert_asset(Asset(asset_id="P-01", location="Plant A"))
    sha, ext = s.store_image(JPEG_1PX)
    s.create_inspection(Inspection(
        inspection_id="INS-1", asset_id="P-01", inspection_date="2026-11-05",
        image_sha256=sha, image_ext=ext,
    ))
    s.add_findings([Finding(
        finding_id="F-1", inspection_id="INS-1", detector_id="x",
        defect_category="crack", bbox=None, confidence=None,
        evidence_description="", limitations=[],
    )])
    s.review_finding("F-1", "confirmed", notes="visible crack", reviewer_name="Admin")
    got = s.list_findings("INS-1")[0]
    assert got.human_review_status == "confirmed"
    assert got.review_notes == "visible crack"
    assert got.reviewed_at is not None


def test_persistence_across_reopen(tmp_path):
    s1 = WorkspaceStore(root=str(tmp_path))
    s1.upsert_asset(Asset(asset_id="P-01", location="Plant A"))
    s1.close()
    s2 = WorkspaceStore(root=str(tmp_path))
    assert s2.get_asset("P-01") is not None