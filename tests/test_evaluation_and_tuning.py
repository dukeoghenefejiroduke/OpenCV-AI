"""Unit tests for continuous crack AUC evaluation, leave-groups-out tuning harness, and field validation manifest / evaluator."""

import os
import json
import csv
import pytest
import numpy as np
from tools.evaluate_tubers_auc import compute_auc
from tools.tune_crack_detector import run_tuning_harness
from tools.field_validation_manifest import generate_manifest_and_template, evaluate_field_dataset, compute_file_sha256


def test_compute_auc_perfect():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(y_true, y_scores) == 1.0


def test_compute_auc_random():
    y_true = [0, 1, 1, 0]
    y_scores = [0.1, 0.2, 0.3, 0.4]
    assert compute_auc(y_true, y_scores) == 0.5


def test_tuning_harness(tmp_path):
    report_path = tmp_path / "tuning_report.json"
    run_tuning_harness(str(report_path))
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["lockbox_groups_count"] >= 4
    assert len(data["cv_folds"]) > 0
    assert "lockbox_evaluation" in data


def test_field_validation_manifest_and_evaluator(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test_pipe.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100) # dummy jpg bytes

    manifest_p, csv_p = generate_manifest_and_template(str(img_dir), str(tmp_path))
    assert os.path.exists(manifest_p)
    assert os.path.exists(csv_p)

    # Populate CSV with agreement
    rows = []
    with open(csv_p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_a"] = "corrosion"
            row["label_b"] = "corrosion"
            rows.append(row)

    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records = evaluate_field_dataset(manifest_p, csv_p)
    assert len(records) == 1
    assert records[0]["final_label"] == "corrosion"

    # Test disagreement without adjudicated label raises ValueError
    rows[0]["label_b"] = "crack"
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Labeler disagreement"):
        evaluate_field_dataset(manifest_p, csv_p)

    # Test disagreement WITH adjudicated label succeeds
    rows[0]["adjudicated_label"] = "corrosion"
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records = evaluate_field_dataset(manifest_p, csv_p)
    assert records[0]["final_label"] == "corrosion"

    # Test manifest hash change raises ValueError
    with open(manifest_p, 'w', encoding='utf-8') as f:
        json.dump({str(img_path): "invalidhash123"}, f)

    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        evaluate_field_dataset(manifest_p, csv_p)
