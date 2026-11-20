import os
import json
import cv2
import numpy as np
from tools.validate_dataset import validate_manifest
from vision.corrosion_detector import inspect_corrosion
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector

def evaluate_expansion():
    manifest_path = "data/field_validation/expanded/expansion_manifest.jsonl"
    errors, warnings, records = validate_manifest(manifest_path)
    if errors:
        print(f"Manifest validation errors: {errors}")
        return

    detectors = {
        "corrosion": None, # uses inspect_corrosion
        "crack": CrackDetector(),
        "surface_defect": SurfaceDefectDetector(),
        "leakage": LeakageDetector(),
        "insulation": InsulationDetector()
    }

    results_by_cat = {cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for cat in detectors}
    detailed_results = []

    for rec in records:
        img_id = rec["image_id"]
        img_path = rec["image_path"]
        haz = rec["hazard_category"]
        gt_label = rec["label"]
        is_gt_positive = (gt_label == "positive")

        img = cv2.imread(img_path)
        if img is None:
            continue

        is_pred_positive = False
        if haz == "corrosion":
            rep = inspect_corrosion(img)
            pct = float(rep.get("corrosion_percentage", 0.0))
            is_pred_positive = pct >= 2.0
        else:
            detector = detectors.get(haz)
            if detector:
                res = detector.detect(img)
                findings = res.get("findings", [])
                is_pred_positive = len(findings) > 0

        if is_gt_positive and is_pred_positive:
            results_by_cat[haz]["tp"] += 1
            pred_str = "positive"
        elif is_gt_positive and not is_pred_positive:
            results_by_cat[haz]["fn"] += 1
            pred_str = "negative"
        elif not is_gt_positive and is_pred_positive:
            results_by_cat[haz]["fp"] += 1
            pred_str = "positive"
        else:
            results_by_cat[haz]["tn"] += 1
            pred_str = "negative"

        detailed_results.append({
            "image_id": img_id,
            "hazard_category": haz,
            "ground_truth": gt_label,
            "prediction": pred_str
        })

    summary_report = {}
    print("\n=== Validation Expansion (100 Images) Evaluation Report ===")
    for cat, counts in results_by_cat.items():
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        total = tp + fp + tn + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        summary_report[cat] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total": total,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4)
        }
        print(f"[{cat.upper()}] (n={total}) -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn} | Precision: {prec:.2f}, Recall: {rec:.2f}, Specificity: {spec:.2f}, F1: {f1:.2f}")

    os.makedirs("data/outputs", exist_ok=True)
    report_path = "data/outputs/expansion_evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary_report, "details": detailed_results}, f, indent=2)
    print(f"\nEvaluation report saved to {report_path}")

if __name__ == "__main__":
    evaluate_expansion()
