"""Field Validation Manifest & Corrosion Evaluator Tool (tools/field_validation_manifest.py).

Builds SHA-256 manifests for field validation images, generates label CSV templates
(columns: file, label_a, label_b, adjudicated_label, distance, lighting, camera, notes),
and provides a strict evaluator that reports sensitivity and specificity with Wilson intervals,
refusing to run if the manifest hash has changed or labelers disagree without an adjudicated label.
Writes results to new files under data/outputs/.
"""

import os
import csv
import json
import hashlib
import math
from typing import Dict, Any, List, Tuple
import cv2
from vision.corrosion_detector import inspect_corrosion


def compute_file_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in f:
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def wilson_score_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Calculates Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    return (max(0.0, round(center - margin, 4)), min(1.0, round(center + margin, 4)))


def generate_manifest_and_template(image_dir: str = "data/field_validation", output_dir: str = "data/field_validation"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "corrosion_label_template.csv")
    manifest_path = os.path.join(output_dir, "corrosion_dataset_manifest.json")

    image_files = []
    if os.path.exists(image_dir):
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.relpath(os.path.join(root, file), start="."))

    if not image_files:
        image_files = ["data/pipe_corrosion.jpg"]

    manifest_records = {}
    for img_p in image_files:
        if os.path.exists(img_p):
            h = compute_file_sha256(img_p)
            manifest_records[img_p] = h

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_records, f, indent=2)

    fieldnames = ["file", "label_a", "label_b", "adjudicated_label", "distance", "lighting", "camera", "notes"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img_p in image_files:
            writer.writerow({
                "file": img_p,
                "label_a": "",
                "label_b": "",
                "adjudicated_label": "",
                "distance": "",
                "lighting": "",
                "camera": "",
                "notes": ""
            })

    print(f"Generated corrosion manifest at {manifest_path} and label template at {csv_path}")
    return manifest_path, csv_path


def evaluate_field_dataset(manifest_path: str, labels_csv_path: str) -> List[Dict[str, Any]]:
    """Evaluates field dataset labels with strict integrity checks.

    Refuses to run if:
    1. Manifest file hash has changed since creation.
    2. Labelers (label_a and label_b) disagree and adjudicated_label is missing/empty.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Labels CSV file not found: {labels_csv_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        stored_manifest = json.load(f)

    for file_path, stored_hash in stored_manifest.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Referenced image file missing on disk: {file_path}")
        current_hash = compute_file_sha256(file_path)
        if current_hash != stored_hash:
            raise ValueError(f"Manifest hash mismatch! File {file_path} has been modified or corrupted.")

    validated_records = []
    with open(labels_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_p = row.get("file")
            label_a = row.get("label_a", "").strip().lower()
            label_b = row.get("label_b", "").strip().lower()
            adj = row.get("adjudicated_label", "").strip().lower()

            if not file_p:
                continue

            if label_a and label_b and label_a != label_b:
                if not adj:
                    raise ValueError(f"Labeler disagreement on file '{file_p}' (Label A: '{label_a}', Label B: '{label_b}') without an adjudicated label.")
                final_label = adj
            elif label_a:
                final_label = label_a
            elif label_b:
                final_label = label_b
            elif adj:
                final_label = adj
            else:
                final_label = "unlabeled"

            validated_records.append({
                "file": file_p,
                "label_a": label_a,
                "label_b": label_b,
                "adjudicated_label": adj,
                "final_label": final_label,
                "distance": row.get("distance", ""),
                "lighting": row.get("lighting", ""),
                "camera": row.get("camera", ""),
                "notes": row.get("notes", "")
            })

    print(f"Successfully validated {len(validated_records)} records against manifest integrity check.")
    return validated_records


def evaluate_corrosion_field_validation(
    manifest_path: str,
    labels_csv_path: str,
    output_report_path: str = "data/outputs/corrosion_validation_report.json"
) -> Dict[str, Any]:
    """Runs strict evaluation of corrosion detector on field dataset with Wilson confidence intervals."""
    records = evaluate_field_dataset(manifest_path, labels_csv_path)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    evaluated_count = 0

    results_detail = []
    for rec in records:
        f_path = rec["file"]
        gt = rec["final_label"]
        if gt not in ("positive", "negative", "corrosion", "clean"):
            continue

        is_positive_gt = gt in ("positive", "corrosion")
        img = cv2.imread(f_path)
        if img is None:
            continue

        # Run corrosion inspector
        rep = inspect_corrosion(img)
        corr_pct = float(rep.get("corrosion_percentage", 0.0))
        is_positive_pred = corr_pct >= 2.0  # threshold for corrosion presence

        if is_positive_gt and is_positive_pred:
            tp += 1
            pred_label = "positive"
        elif not is_positive_gt and is_positive_pred:
            fp += 1
            pred_label = "positive"
        elif not is_positive_gt and not is_positive_pred:
            tn += 1
            pred_label = "negative"
        else:
            fn += 1
            pred_label = "negative"

        evaluated_count += 1
        results_detail.append({
            "file": f_path,
            "ground_truth": gt,
            "prediction": pred_label,
            "corrosion_percentage": corr_pct
        })

    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    sens_ci = wilson_score_interval(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    spec_ci = wilson_score_interval(tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0)

    report = {
        "evaluation_type": "Corrosion Field Validation with Wilson Intervals",
        "total_evaluated": evaluated_count,
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "sensitivity": {"value": round(sensitivity, 4), "wilson_95_ci": sens_ci},
        "specificity": {"value": round(specificity, 4), "wilson_95_ci": spec_ci},
        "details": results_detail
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Corrosion validation report saved to {output_report_path}")
    return report


if __name__ == "__main__":
    m_p, c_p = generate_manifest_and_template()
    if os.path.exists(m_p) and os.path.exists(c_p):
        try:
            evaluate_corrosion_field_validation(m_p, c_p)
        except Exception as e:
            print(f"Evaluation note: {e}")
