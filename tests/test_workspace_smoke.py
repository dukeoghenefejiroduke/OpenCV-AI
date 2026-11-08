import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, Finding, MaintenanceRecord
from workspace.prioritization import score as prioritize
from workspace.report import generate_html_report


def test_workspace_full_workflow_smoke(tmp_path):
    db_root = str(tmp_path / "workspace")
    store = WorkspaceStore(root=db_root)
    try:
        # 1. Create asset
        asset = Asset(asset_id="PIPE-SMOKE-01", location="Test Plant", criticality="high", pipe_material="steel", pipe_type="drain")
        store.upsert_asset(asset)

        # 2. Store valid 1x1 JPEG bytes (smallest valid JPEG magic + EOI)
        valid_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f\xff\xd9"
        sha256, ext = store.store_image(valid_jpeg)

        # 3. Create inspection
        insp = Inspection(
            inspection_id="INS-SMOKE-01",
            asset_id="PIPE-SMOKE-01",
            inspection_date="2026-11-07",
            image_sha256=sha256,
            image_ext=ext,
            detector_results={},
            authoritative_report={"corrosion_percentage": 5.0},
        )
        store.create_inspection(insp)

        # 4. Add findings (one confirmed, one rejected)
        f_conf = Finding(
            finding_id="F-CONF-01", inspection_id="INS-SMOKE-01",
            detector_id="crack_detector", defect_category="crack",
            bbox=[10, 10, 50, 50], confidence=0.85,
            evidence_description="Visible surface crack", limitations=[],
            human_review_status="confirmed"
        )
        f_rej = Finding(
            finding_id="F-REJ-01", inspection_id="INS-SMOKE-01",
            detector_id="leakage_detector", defect_category="leakage",
            bbox=[100, 100, 120, 120], confidence=0.30,
            evidence_description="False positive glare", limitations=[],
            human_review_status="rejected"
        )
        store.add_findings([f_conf, f_rej])

        # 5. Prioritize
        findings = store.list_findings("INS-SMOKE-01")
        history = store.list_inspections_for_asset("PIPE-SMOKE-01")
        priority = prioritize(asset, insp, findings, history)
        assert priority.category != "urgent_human_assessment"
        assert any("rejected" in r.lower() for r in priority.rationale)

        # 6. Generate HTML report
        out_report = str(tmp_path / "reports" / "report.html")
        generate_html_report(store, insp, findings, priority, out_report)
        assert os.path.exists(out_report)
        with open(out_report, "r", encoding="utf-8") as f:
            html_content = f.read()
        assert "Not a certification" in html_content
        assert "crack" in html_content
        assert "leakage" in html_content

    finally:
        store.close()

    # 7. Reopen database and verify persistence
    store2 = WorkspaceStore(root=db_root)
    try:
        fetched_asset = store2.get_asset("PIPE-SMOKE-01")
        assert fetched_asset is not None
        assert fetched_asset.criticality == "high"

        fetched_insp = store2.get_inspection("INS-SMOKE-01")
        assert fetched_insp is not None

        fetched_findings = store2.list_findings("INS-SMOKE-01")
        assert len(fetched_findings) == 2
        statuses = {f.finding_id: f.human_review_status for f in fetched_findings}
        assert statuses["F-CONF-01"] == "confirmed"
        assert statuses["F-REJ-01"] == "rejected"
    finally:
        store2.close()


def test_asset_qr_and_repair_verification(tmp_path):
    db_root = str(tmp_path / "workspace_qr")
    store = WorkspaceStore(root=db_root)
    try:
        # 1. Asset with QR payload
        asset = Asset(
            asset_id="PIPE-QR-01",
            location="Zone B",
            qr_payload="CEMENTSIGHT://ASSET/PIPE-QR-01?mat=steel&crit=critical"
        )
        store.upsert_asset(asset)
        fetched = store.get_asset("PIPE-QR-01")
        assert fetched.qr_payload == "CEMENTSIGHT://ASSET/PIPE-QR-01?mat=steel&crit=critical"

        # 2. Initial inspection & maintenance record / repair verification lifecycle
        sha, ext = store.store_image(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        insp_initial = Inspection(
            inspection_id="INS-INIT-01", asset_id="PIPE-QR-01",
            inspection_date="2026-11-01", image_sha256=sha, image_ext=ext
        )
        store.create_inspection(insp_initial)

        rec = MaintenanceRecord(
            record_id="MR-01",
            asset_id="PIPE-QR-01",
            inspection_id="INS-INIT-01",
            work_performed="Replaced corroded pipe segment",
            performed_by="Technician Jane",
            performed_at="2026-11-03T10:00:00",
            kind="recorded_work",
            verification_status="verified_resolved",
            followup_inspection_id="INS-FOLLOW-01"
        )
        store.add_maintenance_record(rec)

        records = store.list_maintenance_records("PIPE-QR-01")
        assert len(records) == 1
        assert records[0].verification_status == "verified_resolved"
        assert records[0].followup_inspection_id == "INS-FOLLOW-01"
    finally:
        store.close()


def test_qr_parsing_and_badge_generation():
    from tools.server import _parse_asset_id_from_qr, generate_asset_qr_svg
    payload = "CEMENTSIGHT://ASSET/PIPE-QR-01?mat=steel&crit=critical"
    extracted_id = _parse_asset_id_from_qr(payload)
    assert extracted_id == "PIPE-QR-01"

    asset = Asset(asset_id="PIPE-QR-01", location="Zone B", qr_payload=payload, criticality="critical", pipe_material="steel")
    badge_html = generate_asset_qr_svg(asset)
    assert "CementSight Asset Tag" in badge_html
    assert "PIPE-QR-01" in badge_html
    assert "critical" in badge_html
