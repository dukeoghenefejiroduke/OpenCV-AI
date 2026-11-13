"""TUBERS Continuous Crack Scoring & Group-Level AUC Report (tools/evaluate_tubers_auc.py).

Computes a continuous crack score per image (crack contour area / total image pixels),
calculates ROC-AUC at the dataset and group level with cluster bootstrap confidence intervals,
and writes the report to data/outputs/tubers_auc_report.json.
"""

import os
import zipfile
import json
import tempfile
import numpy as np
import cv2
from vision.crack_detector import CrackDetector


def compute_auc(y_true, y_scores):
    """Computes ROC-AUC using numpy without requiring sklearn."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Sort scores descending
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    # Distinct thresholds
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    total_pos = np.sum(y_true_sorted == 1)
    total_neg = np.sum(y_true_sorted == 0)

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = tps / total_pos
    fpr = fps / total_neg

    # Trapezoidal rule (compatible with NumPy 1.x and 2.x)
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)
    return float(auc)


def evaluate_tubers_auc(zip_path: str, output_json_path: str, n_bootstrap: int = 500):
    print(f"Loading TUBERS dataset for continuous scoring & AUC from {zip_path}...")
    if not os.path.exists(zip_path):
        print(f"Archive not found at {zip_path}.")
        return

    detector = CrackDetector()
    groups = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        extracted_root = os.path.join(temp_dir, "Robot Inspection Visual Data")
        if not os.path.exists(extracted_root):
            subdirs = os.listdir(temp_dir)
            if subdirs:
                extracted_root = os.path.join(temp_dir, subdirs[0])

        for root, dirs, files in os.walk(extracted_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = 1 if "crack" in root.lower() and "no-crack" not in root.lower() else 0
                    base_name = os.path.splitext(file)[0]
                    parts = base_name.rsplit('_', 1)
                    prefix = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else base_name

                    if prefix not in groups:
                        groups[prefix] = []

                    img_path = os.path.join(root, file)
                    groups[prefix].append((img_path, label))

        print(f"Scoring {len(groups)} capture session groups...")

        group_records = []
        all_y_true = []
        all_y_scores = []

        for prefix, items in groups.items():
            g_true = []
            g_scores = []
            for img_p, label in items:
                img = cv2.imread(img_p)
                if img is not None:
                    height, width = img.shape[:2]
                    total_pixels = height * width
                    # Run detector to get findings / continuous score
                    res = detector.detect(img)
                    # Continuous score: sum of contour area pixels across all findings divided by total pixels
                    findings = res.get("findings", [])
                    crack_area = sum(f.get("pixel_measurement", {}).get("contour_area_pixels", 0) for f in findings)
                    score = float(crack_area) / float(total_pixels) if total_pixels > 0 else 0.0

                    g_true.append(label)
                    g_scores.append(score)
                    all_y_true.append(label)
                    all_y_scores.append(score)

            g_auc = compute_auc(g_true, g_scores) if len(set(g_true)) > 1 else 0.5
            group_records.append({
                "prefix": prefix,
                "sample_count": len(items),
                "auc": round(g_auc, 4),
                "y_true": g_true,
                "y_scores": g_scores
            })

        overall_auc = compute_auc(all_y_true, all_y_scores)

        # Cluster bootstrap over groups
        np.random.seed(42)
        boot_aucs = []
        if group_records:
            for _ in range(n_bootstrap):
                sampled_groups = np.random.choice(group_records, size=len(group_records), replace=True)
                b_true = []
                b_scores = []
                for g in sampled_groups:
                    b_true.extend(g["y_true"])
                    b_scores.extend(g["y_scores"])
                if len(set(b_true)) > 1:
                    boot_aucs.append(compute_auc(b_true, b_scores))

        ci_low = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else 0.5
        ci_high = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else 0.5

        # Remove raw y_true / y_scores from group_records for clean JSON output
        clean_group_results = [{k: v for k, v in g.items() if k not in ("y_true", "y_scores")} for g in group_records]

        report = {
            "domain_notice": "exploratory, water-pipe domain, not field performance",
            "detector_id": detector.detector_id,
            "detector_version": detector.version,
            "overall_roc_auc": round(overall_auc, 4),
            "cluster_bootstrap_95_ci": {
                "auc_mean": round(float(np.mean(boot_aucs)), 4) if boot_aucs else round(overall_auc, 4),
                "auc_ci_low": round(ci_low, 4),
                "auc_ci_high": round(ci_high, 4)
            },
            "groups_evaluated": len(clean_group_results),
            "group_results": clean_group_results
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"TUBERS continuous AUC report saved to {output_json_path}")
        print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    zip_p = "/data/data/com.termux/files/home/storage/downloads/Robot Inspection Visual Data.zip"
    out_p = "data/outputs/tubers_auc_report.json"
    evaluate_tubers_auc(zip_p, out_p)
