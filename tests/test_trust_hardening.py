"""Comprehensive unit tests for CementSight trust and validation hardening features."""

import os
import json
import csv
import pytest
from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, Finding
from workspace.prioritization import score
from workspace.report import generate_html_report
from tools.field_validation_manifest import (
    generate_manifest_and_template,
    evaluate_field_dataset,
    evaluate_corrosion_field_validation,
    wilson_score_interval
)
from app.forecasting import calculate_degradation_velocity


def test_crack_priority_flag_disabled(tmp_path):
    store = WorkspaceStore(root=str(tmp_path / "ws"))
    asset = Asset(asset_id="PIPE-01", location="Plant A", criticality="critical")
    store.upsert_asset(asset)
    
    sha, ext = store.store_image(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50)
    insp = Inspection(
        inspection_id="INS-1",
        asset_id="PIPE-01",
        inspection_date="2026-11-13",
        image_sha256=sha,
        image_ext=ext
    )
    store.create_inspection(insp)

    finding = Finding(
        finding_id="F-crack",
        inspection_id="INS-1",
        detector_id="crack_screening_detector",
        defect_category="crack",
        bbox=[0, 0, 10, 10],
        confidence=0.8,
        evidence_description="Possible crack",
        limitations=[],
        human_review_status="confirmed"
    )
    store.add_findings([finding])
    findings = store.list_findings("INS-1")

    # With crack_enabled_for_priority=False (default), confirmed crack should NOT add priority points
    p_score_disabled = score(asset, insp, findings, [], crack_enabled_for_priority=False)
    # With crack_enabled_for_priority=True, confirmed crack adds points
    p_score_enabled = score(asset, insp, findings, [], crack_enabled_for_priority=True)

    assert p_score_disabled.score < p_score_enabled.score
    store.close()


def test_review_workflow_requires_reviewer(tmp_path):
    store = WorkspaceStore(root=str(tmp_path / "ws"))
    asset = Asset(asset_id="PIPE-01", location="Plant A")
    store.upsert_asset(asset)
    sha, ext = store.store_image(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50)
    insp = Inspection(inspection_id="INS-1", asset_id="PIPE-01", inspection_date="2026-11-13", image_sha256=sha, image_ext=ext)
    store.create_inspection(insp)

    finding = Finding(
        finding_id="F-1",
        inspection_id="INS-1",
        detector_id="corr",
        defect_category="corrosion",
        bbox=None,
        confidence=0.9,
        evidence_description="Corrosion",
        limitations=[],
        human_review_status="unreviewed"
    )
    store.add_findings([finding])

    # Review without reviewer name must raise ValueError
    with pytest.raises(ValueError, match="Reviewer name is required"):
        store.review_finding("F-1", "confirmed", notes="Looks good", reviewer_name="")

    # Review with reviewer name succeeds and writes audit log
    store.review_finding("F-1", "confirmed", notes="Verified by lead", reviewer_name="Alice")
    findings = store.list_findings("INS-1")
    assert findings[0].human_review_status == "confirmed"

    audit_path = os.path.join(str(tmp_path / "ws"), "review_audit.jsonl")
    assert os.path.exists(audit_path)
    audit_lines = open(audit_path, "r", encoding="utf-8").readlines()
    assert len(audit_lines) == 1
    log_record = json.loads(audit_lines[0])
    assert log_record["reviewer_name"] == "Alice"
    assert log_record["new_state"] == "confirmed"
    store.close()


def test_asset_history_insufficient_observations():
    # Fewer than 3 observations should return 0.0 velocity ("insufficient history")
    history_2 = [(0.0, 1.0), (1.0, 2.0)]
    assert calculate_degradation_velocity(history_2) == 0.0

    history_3 = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert calculate_degradation_velocity(history_3) > 0.0


def test_wilson_score_interval():
    low, high = wilson_score_interval(10, 10)
    assert low >= 0.0
    assert high <= 1.0
    assert low <= high


def test_corrosion_field_validation_evaluator(tmp_path):
    import cv2
    import numpy as np
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "pipe_test.jpg"
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    manifest_p, csv_p = generate_manifest_and_template(str(img_dir), str(tmp_path))

    # Populate CSV
    rows = []
    with open(csv_p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_a"] = "positive"
            row["label_b"] = "positive"
            rows.append(row)

    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = tmp_path / "corrosion_report.json"
    report = evaluate_corrosion_field_validation(manifest_p, csv_p, str(report_path))
    assert report_path.exists()
    assert report["total_evaluated"] == 1
    assert "sensitivity" in report
    assert "specificity" in report
