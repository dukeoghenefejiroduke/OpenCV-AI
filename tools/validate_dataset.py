#!/usr/bin/env python3
"""Dataset validation utility for CementSight AI v1.3.0 / v1.4.0.

Validates dataset manifests (JSONL) and annotation schema records against quality,
provenance, bounding-box validity, split leakage, eligibility, and annotation status rules.
"""

import os
import sys
import json

VALID_PROVENANCES = {"real", "synthetic", "generated", "fixture"}
VALID_HAZARD_CATEGORIES = {
    "corrosion",
    "crack",
    "surface_defect",
    "leakage",
    "insulation",
    "clean_negative",
}
VALID_ANNOTATION_STATUSES = {
    "annotated",
    "unreviewed",
    "ambiguous",
    "insufficient_evidence",
}
VALID_LABELS = {"positive", "negative", "ambiguous"}
VALID_SPLITS = {"development", "validation", "held_out_test"}
REQUIRED_FIELDS = [
    "image_id",
    "image_path",
    "provenance",
    "hazard_category",
    "annotation_status",
    "label",
    "dataset_split",
    "reviewer_id",
    "annotation_timestamp",
]


def validate_manifest(manifest_path="data/field_validation/manifest.jsonl"):
    """Validates a JSONL manifest file and returns (errors, warnings, records)."""
    errors = []
    warnings = []
    records = []

    if not os.path.exists(manifest_path):
        errors.append(f"Manifest file not found: {manifest_path}")
        return errors, warnings, records

    seen_ids = set()
    group_to_splits = {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Malformed JSON: {e}")
                continue

            if not isinstance(record, dict):
                errors.append(f"Line {line_num}: Record must be a JSON object.")
                continue

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in record:
                    errors.append(f"Line {line_num} ({record.get('image_id', 'UNKNOWN')}): Missing required field '{field}'")

            image_id = record.get("image_id")
            if image_id:
                if image_id in seen_ids:
                    errors.append(f"Line {line_num}: Duplicate image_id found: '{image_id}'")
                else:
                    seen_ids.add(image_id)

            # Validate provenance
            prov = record.get("provenance")
            if prov and prov not in VALID_PROVENANCES:
                errors.append(f"Line {line_num} ({image_id}): Invalid provenance '{prov}'. Must be one of {VALID_PROVENANCES}")

            eligible = record.get("eligible_for_real_world_evaluation", False)
            if eligible:
                if prov != "real":
                    errors.append(f"Line {line_num} ({image_id}): Record claims real-world evaluation eligibility but provenance is '{prov}' (must be 'real').")
                if not record.get("source_url") or not record.get("license") or not record.get("attribution"):
                    errors.append(f"Line {line_num} ({image_id}): Record claims real-world evaluation eligibility but lacks required source_url, license, or attribution metadata.")
            else:
                if prov == "real":
                    warnings.append(f"Line {line_num} ({image_id}): Provenance is 'real' but record is marked ineligible for real-world evaluation (unverified/fixture).")

            # Validate hazard category
            haz = record.get("hazard_category")
            if haz and haz not in VALID_HAZARD_CATEGORIES:
                errors.append(f"Line {line_num} ({image_id}): Invalid hazard_category '{haz}'. Must be one of {VALID_HAZARD_CATEGORIES}")

            # Validate annotation status
            status = record.get("annotation_status")
            if status and status not in VALID_ANNOTATION_STATUSES:
                errors.append(f"Line {line_num} ({image_id}): Invalid annotation_status '{status}'. Must be one of {VALID_ANNOTATION_STATUSES}")

            # Validate label
            lbl = record.get("label")
            if lbl and lbl not in VALID_LABELS:
                errors.append(f"Line {line_num} ({image_id}): Invalid label '{lbl}'. Must be one of {VALID_LABELS}")

            # Unreviewed check: unreviewed images must not be treated as confirmed negatives
            if status == "unreviewed" and lbl == "negative":
                warnings.append(f"Line {line_num} ({image_id}): Unreviewed image is marked as negative. Unreviewed images should not be treated as confirmed negatives.")

            # Validate dataset split
            split = record.get("dataset_split")
            if split and split not in VALID_SPLITS:
                errors.append(f"Line {line_num} ({image_id}): Invalid dataset_split '{split}'. Must be one of {VALID_SPLITS}")

            # Track grouping_id for split leakage
            group_id = record.get("grouping_id")
            if group_id and split:
                if group_id not in group_to_splits:
                    group_to_splits[group_id] = set()
                group_to_splits[group_id].add(split)

            # Validate image path existence
            img_path = record.get("image_path")
            if img_path:
                if not os.path.exists(img_path):
                    errors.append(f"Line {line_num} ({image_id}): Image file path does not exist on disk: '{img_path}'")

            # Validate image dimensions & bounding boxes
            dims = record.get("image_dimensions")
            width, height = None, None
            if dims:
                if not isinstance(dims, dict) or "width" not in dims or "height" not in dims:
                    errors.append(f"Line {line_num} ({image_id}): 'image_dimensions' must be an object with 'width' and 'height'")
                else:
                    width = dims.get("width")
                    height = dims.get("height")

            bboxes = record.get("bounding_boxes", [])
            if not isinstance(bboxes, list):
                errors.append(f"Line {line_num} ({image_id}): 'bounding_boxes' must be a list")
            else:
                for idx, bbox in enumerate(bboxes):
                    if not isinstance(bbox, dict):
                        errors.append(f"Line {line_num} ({image_id}), bbox {idx}: Bounding box must be an object")
                        continue
                    for bf in ["x_min", "y_min", "x_max", "y_max", "defect_type"]:
                        if bf not in bbox:
                            errors.append(f"Line {line_num} ({image_id}), bbox {idx}: Missing bounding box field '{bf}'")
                    
                    x_min = bbox.get("x_min", 0)
                    y_min = bbox.get("y_min", 0)
                    x_max = bbox.get("x_max", 0)
                    y_max = bbox.get("y_max", 0)

                    if x_min >= x_max or y_min >= y_max:
                        errors.append(f"Line {line_num} ({image_id}), bbox {idx}: Invalid coordinates (x_min >= x_max or y_min >= y_max)")

                    if width is not None and height is not None:
                        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                            errors.append(f"Line {line_num} ({image_id}), bbox {idx}: Bounding box [{x_min}, {y_min}, {x_max}, {y_max}] is out of bounds for image dimensions {width}x{height}")

            records.append(record)

    # Check split leakage from grouping IDs
    for group_id, splits in group_to_splits.items():
        if len(splits) > 1:
            errors.append(f"Split leakage detected: grouping_id '{group_id}' spans multiple dataset splits: {splits}")

    return errors, warnings, records


def main():
    manifest_path = "data/field_validation/manifest.jsonl"
    print(f"Validating dataset manifest: {manifest_path}")
    errors, warnings, records = validate_manifest(manifest_path)

    print(f"Total records validated: {len(records)}")
    if warnings:
        print("\n[WARNINGS]:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n[ERRORS]:")
        for e in errors:
            print(f"  - {e}")
        print("\nDataset validation FAILED.")
        sys.exit(1)
    else:
        print("\nDataset validation PASSED successfully with no errors.")
        sys.exit(0)


if __name__ == "__main__":
    main()
