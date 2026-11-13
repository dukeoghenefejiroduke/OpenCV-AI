"""Unit tests for Continuous AUC, Leave-Groups-Out Tuning Harness, and Field Validation Manifest Tool (tests/test_cross_hazard_benchmark.py)."""

import os
import json
import csv
import pytest
import numpy as np

from tools.evaluate_tubers_auc import compute_auc
from tools.tune_crack_detector import run_tuning_harness
from tools.field_validation_manifest import (
    compute_file_sha256,
    generate_manifest_and_template,
    evaluate_field_dataset
)


def test_compute_auc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auc = compute_auc(y_true, y_scores)
    assert 0.0 <= auc <= 1.0


def test_tuning_harness_lockbox(tmp_path):
    report_path = tmp_path / "tuning_report.json"
    run_tuning_harness(str(report_path))
    assert os.path.exists(report_path)
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data["lockbox_groups_count"] >= 4
    assert len(data["lockbox_prefixes"]) >= 4


def test_field_validation_manifest_and_evaluator(tmp_path):
    # Setup test image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_file = img_dir / "sample_pipe.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100) # dummy jpg bytes

    manifest_p, csv_p = generate_manifest_and_template(image_dir=str(img_dir), output_dir=str(tmp_path))
    assert os.path.exists(manifest_p)
    assert os.path.exists(csv_p)

    # Populate CSV with agreeing labels
    rows = []
    with open(csv_p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row["label_a"] = "positive"
            row["label_b"] = "positive"
            rows.append(row)

    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records = evaluate_field_dataset(manifest_p, csv_p)
    assert len(records) == 1
    assert records[0]["final_label"] == "positive"


def test_field_validation_disagreement_raises_error(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_file = img_dir / "sample_pipe.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100)

    manifest_p, csv_p = generate_manifest_and_template(image_dir=str(img_dir), output_dir=str(tmp_path))

    rows = []
    with open(csv_p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row["label_a"] = "positive"
            row["label_b"] = "negative" # Disagreement without adjudication
            rows.append(row)

    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Labeler disagreement"):
        evaluate_field_dataset(manifest_p, csv_p)
