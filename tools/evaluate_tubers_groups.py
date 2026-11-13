"""TUBERS Group-Level Exploratory Evaluation Script (tools/evaluate_tubers_groups.py).

Performs group-level exploratory evaluation of the classical crack detector on the TUBERS dataset,
grouped by capture session prefix (rather than official partitions) with cluster-bootstrap confidence intervals.
Explicitly labelled: 'exploratory, water-pipe domain, not field performance'.
"""

import os
import zipfile
import json
import tempfile
import numpy as np
import cv2
from vision.crack_detector import CrackDetector


def evaluate_tubers_groups(zip_path: str, output_json_path: str, n_bootstrap: int = 500):
    print(f"Loading TUBERS dataset from {zip_path}...")
    if not os.path.exists(zip_path):
        print(f"Archive not found at {zip_path}. Skipping group evaluation.")
        return

    detector = CrackDetector()
    groups = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        # Find labels.csv or traverse directories
        extracted_root = os.path.join(temp_dir, "Robot Inspection Visual Data")
        if not os.path.exists(extracted_root):
            # Try finding root
            subdirs = os.listdir(temp_dir)
            if subdirs:
                extracted_root = os.path.join(temp_dir, subdirs[0])

        # Walk through train/val/test or partitions
        for root, dirs, files in os.walk(extracted_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = "crack" if "crack" in root.lower() and "no-crack" not in root.lower() else "no_crack"
                    # Determine prefix group
                    # e.g., WIN_20251120_14_10_36_Pro_000035.jpg -> prefix = WIN_20251120_14_10_36_Pro
                    base_name = os.path.splitext(file)[0]
                    # Strip trailing frame number if underscore followed by digits
                    parts = base_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        prefix = parts[0]
                    else:
                        prefix = base_name

                    if prefix not in groups:
                        groups[prefix] = {"crack_images": [], "no_crack_images": []}

                    img_path = os.path.join(root, file)
                    if label == "crack":
                        groups[prefix]["crack_images"].append(img_path)
                    else:
                        groups[prefix]["no_crack_images"].append(img_path)

        print(f"Found {len(groups)} capture session groups.")

        group_results = []
        for prefix, data in groups.items():
            crack_total = len(data["crack_images"])
            crack_detected = 0
            for img_p in data["crack_images"]:
                img = cv2.imread(img_p)
                if img is not None:
                    res = detector.detect(img)
                    if res.get("execution_status") == "CANDIDATE_DETECTED":
                        crack_detected += 1

            no_crack_total = len(data["no_crack_images"])
            no_crack_fp = 0
            for img_p in data["no_crack_images"]:
                img = cv2.imread(img_p)
                if img is not None:
                    res = detector.detect(img)
                    if res.get("execution_status") == "CANDIDATE_DETECTED":
                        no_crack_fp += 1

            det_rate = (crack_detected / crack_total) if crack_total > 0 else 0.0
            fp_rate = (no_crack_fp / no_crack_total) if no_crack_total > 0 else 0.0

            group_results.append({
                "prefix": prefix,
                "crack_total": crack_total,
                "crack_detected": crack_detected,
                "detection_rate": round(det_rate, 4),
                "no_crack_total": no_crack_total,
                "no_crack_fp": no_crack_fp,
                "false_positive_rate": round(fp_rate, 4)
            })

        # Cluster bootstrap over groups
        np.random.seed(42)
        boot_det_rates = []
        boot_fp_rates = []

        if group_results:
            group_arr = np.array(group_results)
            for _ in range(n_bootstrap):
                sample = np.random.choice(group_results, size=len(group_results), replace=True)
                total_crack = sum(g["crack_total"] for g in sample)
                total_crack_det = sum(g["crack_detected"] for g in sample)
                total_nc = sum(g["no_crack_total"] for g in sample)
                total_nc_fp = sum(g["no_crack_fp"] for g in sample)

                boot_det_rates.append(total_crack_det / total_crack if total_crack > 0 else 0.0)
                boot_fp_rates.append(total_nc_fp / total_nc if total_nc > 0 else 0.0)

        report = {
            "domain_notice": "exploratory, water-pipe domain, not field performance",
            "detector_id": detector.detector_id,
            "detector_version": detector.version,
            "groups_evaluated": len(group_results),
            "group_results": group_results,
            "cluster_bootstrap_95_ci": {
                "detection_rate_mean": round(float(np.mean(boot_det_rates)), 4) if boot_det_rates else 0.0,
                "detection_rate_ci_low": round(float(np.percentile(boot_det_rates, 2.5)), 4) if boot_det_rates else 0.0,
                "detection_rate_ci_high": round(float(np.percentile(boot_det_rates, 97.5)), 4) if boot_det_rates else 0.0,
                "false_positive_rate_mean": round(float(np.mean(boot_fp_rates)), 4) if boot_fp_rates else 0.0,
                "false_positive_rate_ci_low": round(float(np.percentile(boot_fp_rates, 2.5)), 4) if boot_fp_rates else 0.0,
                "false_positive_rate_ci_high": round(float(np.percentile(boot_fp_rates, 97.5)), 4) if boot_fp_rates else 0.0,
            }
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"TUBERS group evaluation report saved to {output_json_path}")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    zip_p = "/data/data/com.termux/files/home/storage/downloads/Robot Inspection Visual Data.zip"
    out_p = "data/outputs/tubers_group_evaluation.json"
    evaluate_tubers_groups(zip_p, out_p)
