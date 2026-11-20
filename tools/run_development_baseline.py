import os
import json
import cv2
from tools.validate_dataset import validate_manifest
from vision.corrosion_detector import inspect_corrosion
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector

def run_baseline():
    manifest_path = "data/development/development_manifest.jsonl"
    errors, warnings, records = validate_manifest(manifest_path)
    print(f"Development Manifest Inspection -> Records: {len(records)}, Errors: {len(errors)}")

    detectors = {
        "corrosion": None,
        "crack": CrackDetector(),
        "surface_defect": SurfaceDefectDetector(),
        "leakage": LeakageDetector(),
        "insulation": InsulationDetector()
    }

    counts = {cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0} for cat in detectors}

    for rec in records:
        haz = rec["hazard_category"]
        gt_positive = (rec["label"] == "positive")
        img_path = rec["image_path"]
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        counts[haz]["total"] += 1
        is_pred_positive = False
        
        if haz == "corrosion":
            rep = inspect_corrosion(img)
            pct = float(rep.get("corrosion_percentage", 0.0))
            is_pred_positive = pct >= 2.0
        else:
            detector = detectors.get(haz)
            if detector:
                res = detector.detect(img)
                is_pred_positive = len(res.get("findings", [])) > 0
                
        if gt_positive and is_pred_positive:
            counts[haz]["tp"] += 1
        elif gt_positive and not is_pred_positive:
            counts[haz]["fn"] += 1
        elif not gt_positive and is_pred_positive:
            counts[haz]["fp"] += 1
        else:
            counts[haz]["tn"] += 1

    print("\n=== Development Baseline Evaluation ===")
    for cat, c in counts.items():
        total = c["total"]
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"[{cat.upper()}] (n={total}) -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn} | Precision: {prec:.2f}, Recall: {rec:.2f}")

if __name__ == "__main__":
    run_baseline()
