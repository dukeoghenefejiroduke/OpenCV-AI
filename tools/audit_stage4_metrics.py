import os
import cv2
import numpy as np
from vision.corrosion_detector import inspect_corrosion
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector

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
    print("=== STAGE 4 AUDIT VERIFICATION SCRIPT ===")
    
    # Batch 1 Corrosion
    b1_items = [
        ("field_01.jpg", True), ("field_02.jpg", True), ("field_03.jpg", True),
        ("field_04.jpg", True), ("field_05.jpg", True), ("field_06.jpg", False),
        ("field_07.jpg", False), ("field_08.jpg", False), ("field_09.jpg", False),
        ("field_10.jpg", False)
    ]
    tp1, fp1, tn1, fn1 = 0, 0, 0, 0
    for fname, gt in b1_items:
        img = cv2.imread(os.path.join("data/field_validation", fname))
        rep = inspect_corrosion(img)
        pred = float(rep.get("corrosion_percentage", 0.0)) >= 2.0
        if gt and pred: tp1 += 1
        elif gt and not pred: fn1 += 1
        elif not gt and pred: fp1 += 1
        elif not gt and not pred: tn1 += 1
    
    # Batch 3 Corrosion
    b3_items = [
        ("batch3_01_pipe_sun.jpg", True), ("batch3_02_pipe_shade.jpg", True),
        ("batch3_03_flange_angle.jpg", True), ("batch3_04_joint_cloudy.jpg", True),
        ("batch3_05_thin_corr.jpg", True), ("batch3_06_clean_gray.jpg", False),
        ("batch3_07_clean_wood.jpg", False), ("batch3_08_clean_crack.jpg", False),
        ("batch3_09_clean_wall.jpg", False), ("batch3_10_clean_orange.jpg", False)
    ]
    tp3, fp3, tn3, fn3 = 0, 0, 0, 0
    for fname, gt in b3_items:
        img = cv2.imread(os.path.join("data/field_validation/batch3", fname))
        rep = inspect_corrosion(img)
        pred = float(rep.get("corrosion_percentage", 0.0)) >= 2.0
        if gt and pred: tp3 += 1
        elif gt and not pred: fn3 += 1
        elif not gt and pred: fp3 += 1
        elif not gt and not pred: tn3 += 1

    # Combined 20-image Corrosion
    tpc, fpc, tnc, fnc = tp1 + tp3, fp1 + fp3, tn1 + tn3, fn1 + fn3

    print("Batch 1 Corrosion Metrics:", calc_metrics(tp1, fp1, tn1, fn1))
    print("Batch 3 Corrosion Metrics:", calc_metrics(tp3, fp3, tn3, fn3))
    print("Combined 20-Image Corrosion Metrics:", calc_metrics(tpc, fpc, tnc, fnc))

    # Batch 2 Multi-Hazard
    b2_items = [
        ("batch2_01_crack.jpg", "crack", True), ("batch2_02_crack.jpg", "crack", True), ("batch2_03_crack.jpg", "crack", False),
        ("batch2_04_coating.jpg", "surface_defect", True), ("batch2_05_coating.jpg", "surface_defect", True), ("batch2_06_coating.jpg", "surface_defect", False),
        ("batch2_07_leakage.jpg", "leakage", True), ("batch2_08_leakage.jpg", "leakage", False),
        ("batch2_09_insulation.jpg", "insulation", True), ("batch2_10_insulation.jpg", "insulation", False)
    ]
    detectors = {
        "crack": CrackDetector(),
        "surface_defect": SurfaceDefectDetector(),
        "leakage": LeakageDetector(),
        "insulation": InsulationDetector()
    }
    b2_counts = {cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for cat in detectors}
    for fname, cat, gt in b2_items:
        img = cv2.imread(os.path.join("data/field_validation/batch2", fname))
        res = detectors[cat].detect(img)
        pred = len(res.get("findings", [])) > 0
        if gt and pred: b2_counts[cat]["tp"] += 1
        elif gt and not pred: b2_counts[cat]["fn"] += 1
        elif not gt and pred: b2_counts[cat]["fp"] += 1
        elif not gt and not pred: b2_counts[cat]["tn"] += 1

    for cat, counts in b2_counts.items():
        print(f"Batch 2 [{cat}] Metrics:", calc_metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"]))

if __name__ == "__main__":
    main()
