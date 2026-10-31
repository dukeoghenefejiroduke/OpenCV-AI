"""Unit tests for CementSight v1.3.0 Phase 1 Dataset & Annotation Framework."""

import os
import json
import pytest
from tools.validate_dataset import validate_manifest


def test_validate_default_manifest():
    """Verify that the default manifest validates successfully without errors."""
    errors, warnings, records = validate_manifest("data/field_validation/manifest.jsonl")
    assert errors == [], f"Unexpected validation errors: {errors}"
    assert len(records) >= 6


def test_manifest_not_found(tmp_path):
    """Verify error handling when manifest file does not exist."""
    missing_file = tmp_path / "non_existent.jsonl"
    errors, warnings, records = validate_manifest(str(missing_file))
    assert len(errors) > 0
    assert "Manifest file not found" in errors[0]


def test_malformed_json_record(tmp_path):
    """Verify error handling for malformed JSON lines."""
    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text("{not valid json\n")
    errors, warnings, records = validate_manifest(str(bad_manifest))
    assert any("Malformed JSON" in e for e in errors)


def test_duplicate_image_ids(tmp_path):
    """Verify detection of duplicate image IDs."""
    dup_manifest = tmp_path / "dup.jsonl"
    rec = {
        "image_id": "test_dup",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    dup_manifest.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(dup_manifest))
    assert any("Duplicate image_id" in e for e in errors)


def test_missing_image_reference(tmp_path):
    """Verify detection of missing image file paths on disk."""
    miss_manifest = tmp_path / "missing.jsonl"
    rec = {
        "image_id": "test_missing",
        "image_path": "data/non_existent_image_12345.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    miss_manifest.write_text(json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(miss_manifest))
    assert any("Image file path does not exist" in e for e in errors)


def test_split_leakage_detection(tmp_path):
    """Verify detection of split leakage when same grouping_id spans multiple splits."""
    leak_manifest = tmp_path / "leak.jsonl"
    rec1 = {
        "image_id": "img_a",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "grouping_id": "shared_group_1",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    rec2 = {
        "image_id": "img_b",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "development",
        "grouping_id": "shared_group_1",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    leak_manifest.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")
    errors, warnings, records = validate_manifest(str(leak_manifest))
    assert any("Split leakage detected" in e for e in errors)


def test_out_of_bounds_bounding_box(tmp_path):
    """Verify detection of out-of-bounds bounding box coordinates."""
    bbox_manifest = tmp_path / "bbox.jsonl"
    rec = {
        "image_id": "img_bbox",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "image_dimensions": {"width": 100, "height": 100},
        "bounding_boxes": [{"x_min": 10, "y_min": 10, "x_max": 150, "y_max": 50, "defect_type": "rust"}],
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    bbox_manifest.write_text(json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(bbox_manifest))
    assert any("out of bounds" in e for e in errors)


def test_unreviewed_warning(tmp_path):
    """Verify warning when unreviewed image is labeled negative."""
    unrev_manifest = tmp_path / "unrev.jsonl"
    rec = {
        "image_id": "img_unrev",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "unreviewed",
        "label": "negative",
        "dataset_split": "validation",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    unrev_manifest.write_text(json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(unrev_manifest))
    assert any("Unreviewed image is marked as negative" in w for w in warnings)


def test_real_provenance_missing_attribution_warning(tmp_path):
    """Verify warning when real provenance record lacks eligibility (unverified/fixture)."""
    prov_manifest = tmp_path / "prov.jsonl"
    rec = {
        "image_id": "img_prov",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    prov_manifest.write_text(json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(prov_manifest))
    assert any("marked ineligible for real-world evaluation" in w for w in warnings)


def test_real_world_eligibility_error(tmp_path):
    """Verify error when record claims real-world evaluation eligibility without required metadata."""
    elig_manifest = tmp_path / "elig.jsonl"
    rec = {
        "image_id": "img_elig",
        "image_path": "data/pipe_corrosion.jpg",
        "provenance": "real",
        "eligible_for_real_world_evaluation": True,
        "hazard_category": "corrosion",
        "annotation_status": "annotated",
        "label": "positive",
        "dataset_split": "validation",
        "reviewer_id": "rev_1",
        "annotation_timestamp": "2026-10-31T00:00:00Z"
    }
    elig_manifest.write_text(json.dumps(rec) + "\n")
    errors, warnings, records = validate_manifest(str(elig_manifest))
    assert any("lacks required source_url, license, or attribution metadata" in e for e in errors)


def test_all_five_hazard_categories_and_provenances():
    """Verify that all five hazard categories plus clean_negative and all provenances are supported."""
    errors, warnings, records = validate_manifest("data/field_validation/manifest.jsonl")
    categories = {r["hazard_category"] for r in records}
    provenances = {r["provenance"] for r in records}

    expected_categories = {"corrosion", "crack", "surface_defect", "leakage", "insulation", "clean_negative"}
    assert expected_categories.issubset(categories)
    assert {"synthetic", "generated", "fixture"}.issubset(provenances)
