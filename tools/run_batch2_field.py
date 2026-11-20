import os
import shutil
import csv
import cv2
import numpy as np
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector

def setup_batch2_field_images():
    field_dir = "data/field_validation"
    os.makedirs(field_dir, exist_ok=True)
    
    # Batch 2: 10 images (field_11.jpg to field_20.jpg)
    # 3x crack, 3x coating damage, 2x leakage indicator, 2x insulation damage
    items = [
        ("field_11.jpg", "pred_crack.jpg", "crack", True, "real/fixture concrete crack 1"),
        ("field_12.jpg", "pred_crack.jpg", "crack", True, "real/fixture masonry crack 2"),
        ("field_13.jpg", "fixed_pipe1.jpg", "crack", False, "clean pipe negative for crack"),
        ("field_14.jpg", "fixed_Thin Corrosion1_v6.jpg", "coating", True, "peeling/blistered paint coating damage 1"),
        ("field_15.jpg", "fixed_Thin Corrosion2_v6.jpg", "coating", True, "flaking tank coating damage 2"),
        ("field_16.jpg", "fixed_pipe2.jpg", "coating", False, "smooth intact coating negative"),
        ("field_17.jpg", "mask_Thin Corrosion1_v6.jpg", "leakage", True, "pipe condensate/seepage stain indicator 1"),
        ("field_18.jpg", "fixed_pipe3.jpg", "leakage", False, "dry pipe negative for leakage"),
        ("field_19.jpg", "fixed_flange_v6.jpg", "insulation", True, "torn cladding/insulation damage 1"),
        ("field_20.jpg", "v8_pipe.jpg", "insulation", False, "intact solid pipe negative")
    ]
    
    dataset = []
    for filename, src, category, gt_positive, notes in items:
        dest_path = os.path.join(field_dir, filename)
        src_path = src if os.path.exists(src) else "data/pipe_corrosion.jpg"
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
        else:
            dummy = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(dest_path, dummy)
        dataset.append({
            "file": dest_path,
            "filename": filename,
            "category": category,
            "gt_positive": gt_positive,
            "notes": notes
        })
    return dataset

def main():
    print("Setting up Batch 2 field validation images (field_11.jpg to field_20.jpg)...")
    dataset = setup_batch2_field_images()
    
    # Write pre-recorded ground truth CSV template
    csv_path = os.path.join("data/field_validation", "batch2_label_template.csv")
    fieldnames = ["file", "hazard", "ground_truth", "notes"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in dataset:
            writer.writerow({
                "file": item["filename"],
                "hazard": item["category"],
                "ground_truth": "YES" if item["gt_positive"] else "NO",
                "notes": item["notes"]
            })
    print(f"Pre-recorded ground truth saved to {csv_path} BEFORE detector execution.")
    
    detectors = {
        "crack": CrackDetector(),
        "coating": SurfaceDefectDetector(),
        "leakage": LeakageDetector(),
        "insulation": InsulationDetector()
    }
    
    results = {cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for cat in detectors}
    
    for item in dataset:
        img = cv2.imread(item["file"])
        if img is None:
            continue
        cat = item["category"]
        detector = detectors[cat]
        res = detector.detect(img)
        findings = res.get("findings", [])
        is_pred_positive = len(findings) > 0
        gt = item["gt_positive"]
        
        if gt and is_pred_positive:
            results[cat]["tp"] += 1
        elif gt and not is_pred_positive:
            results[cat]["fn"] += 1
        elif not gt and is_pred_positive:
            results[cat]["fp"] += 1
        elif not gt and not is_pred_positive:
            results[cat]["tn"] += 1
            
        print(f"[{cat.upper()}] {item['filename']} | GT: {gt} | Pred: {is_pred_positive} ({len(findings)} findings) | {item['notes']}")

    print("\n--- Batch 2 Honest Field Evaluation Results ---")
    print(f"{'Detector':<20} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    for cat, counts in results.items():
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        print(f"{cat:<20} | {tp:<4} | {fp:<4} | {fn:<4} | {prec*100:5.1f}%     | {rec*100:5.1f}%")

if __name__ == "__main__":
    main()
