"""Evaluation script for Mission Two: Comparing Candidates A, B, C, D, and E across the 20-image dataset."""

import os
import cv2
from vision.experimental_detector import ExperimentalCorrosionDetector


def calc_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4)
    }


def main():
    print("=== MISSION TWO CONTROLLED BENCHMARK ===")

    # Define 20-image dataset items with ground truth and material metadata
    # Batch 1 (10 items) and Batch 3 (10 items)
    b1_items = [
        ("field_01.jpg", True, "steel"), ("field_02.jpg", True, "steel"), ("field_03.jpg", True, "steel"),
        ("field_04.jpg", True, "steel"), ("field_05.jpg", True, "steel"), ("field_06.jpg", False, "concrete"),
        ("field_07.jpg", False, "wood"), ("field_08.jpg", False, "steel"), ("field_09.jpg", False, "brick"),
        ("field_10.jpg", False, "steel")
    ]
    b3_items = [
        ("batch3_01_pipe_sun.jpg", True, "steel"), ("batch3_02_pipe_shade.jpg", True, "steel"),
        ("batch3_03_flange_angle.jpg", True, "steel"), ("batch3_04_joint_cloudy.jpg", True, "steel"),
        ("batch3_05_thin_corr.jpg", True, "steel"), ("batch3_06_clean_gray.jpg", False, "steel"),
        ("batch3_07_clean_wood.jpg", False, "wood"), ("batch3_08_clean_crack.jpg", False, "concrete"),
        ("batch3_09_clean_wall.jpg", False, "brick"), ("batch3_10_clean_orange.jpg", False, "unknown")
    ]
    all_items = []
    for fname, gt, mat in b1_items:
        path = os.path.join("data/field_validation", fname)
        all_items.append((path, gt, mat))
    for fname, gt, mat in b3_items:
        path = os.path.join("data/field_validation/batch3", fname)
        all_items.append((path, gt, mat))

    candidates = ["A", "B", "C", "D", "E"]
    results = {}

    for cand in candidates:
        detector = ExperimentalCorrosionDetector(candidate=cand)
        tp, fp, tn, fn = 0, 0, 0, 0
        for path, gt, mat in all_items:
            if not os.path.exists(path):
                continue
            res = detector.inspect(path, material_type=mat)
            pred = res["corrosion_percentage"] >= 2.0
            if gt and pred: tp += 1
            elif gt and not pred: fn += 1
            elif not gt and pred: fp += 1
            elif not gt and not pred: tn += 1

        metrics = calc_metrics(tp, fp, tn, fn)
        results[cand] = metrics
        print(f"Candidate {cand}: {metrics}")

    print("=== END MISSION TWO BENCHMARK ===")


if __name__ == "__main__":
    main()
