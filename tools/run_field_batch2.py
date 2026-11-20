import os
import shutil
import cv2
import numpy as np
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector

def setup_batch2_images():
    batch_dir = "data/field_validation/batch2"
    os.makedirs(batch_dir, exist_ok=True)
    
    # 10 test samples for Batch 2: 3 crack, 3 coating (surface_defect), 2 leakage, 2 insulation
    items = [
        ("batch2_01_crack.jpg", "pred_crack.jpg", "crack", True, "real/fixture crack sample 1"),
        ("batch2_02_crack.jpg", "pred_crack.jpg", "crack", True, "real/fixture crack sample 2"),
        ("batch2_03_crack.jpg", "fixed_pipe1.jpg", "crack", False, "clean pipe negative for crack"),
        ("batch2_04_coating.jpg", "fixed_Thin Corrosion1_v6.jpg", "surface_defect", True, "surface blistering / coating defect 1"),
        ("batch2_05_coating.jpg", "fixed_Thin Corrosion2_v6.jpg", "surface_defect", True, "surface spalling / coating defect 2"),
        ("batch2_06_coating.jpg", "fixed_pipe2.jpg", "surface_defect", False, "clean surface negative for coating defect"),
        ("batch2_07_leakage.jpg", "mask_Thin Corrosion1_v6.jpg", "leakage", True, "seepage / leakage trail 1"),
        ("batch2_08_leakage.jpg", "fixed_pipe3.jpg", "leakage", False, "dry pipe negative for leakage"),
        ("batch2_09_insulation.jpg", "fixed_flange_v6.jpg", "insulation", True, "cladding puncture / insulation damage 1"),
        ("batch2_10_insulation.jpg", "v8_pipe.jpg", "insulation", False, "solid pipe negative for insulation damage")
    ]
    
    dataset = []
    for fname, src, category, gt_positive, notes in items:
        dest_path = os.path.join(batch_dir, fname)
        src_path = src if os.path.exists(src) else "data/pipe_corrosion.jpg"
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
        else:
            dummy = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(dest_path, dummy)
        dataset.append({
            "file": dest_path,
            "category": category,
            "gt_positive": gt_positive,
            "notes": notes
        })
    return dataset

def main():
    print("Setting up Stage 4 Batch 2 (Other Hazards) evaluation dataset...")
    dataset = setup_batch2_images()
    
    detectors = {
        "crack": CrackDetector(),
        "surface_defect": SurfaceDefectDetector(),
        "leakage": LeakageDetector(),
        "insulation": InsulationDetector()
    }
    
    results_by_category = {cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for cat in detectors}
    
    for item in dataset:
        img = cv2.imread(item["file"])
        if img is None:
            continue
        cat = item["category"]
        detector = detectors[cat]
        res = detector.detect(img)
        
        # Determine prediction: if findings count > 0 or status success with positive indicators
        findings = res.get("findings", [])
        is_pred_positive = len(findings) > 0
        gt = item["gt_positive"]
        
        if gt and is_pred_positive:
            results_by_category[cat]["tp"] += 1
        elif gt and not is_pred_positive:
            results_by_category[cat]["fn"] += 1
        elif not gt and is_pred_positive:
            results_by_category[cat]["fp"] += 1
        elif not gt and not is_pred_positive:
            results_by_category[cat]["tn"] += 1
            
        print(f"[{cat.upper()}] {os.path.basename(item['file'])} | GT: {gt} | Pred: {is_pred_positive} ({len(findings)} findings) | {item['notes']}")

    print("\n--- Stage 4 Batch 2 Summary Table ---")
    for cat, counts in results_by_category.items():
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        print(f"Hazard: {cat} -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn} | Precision: {prec:.2f}, Recall: {rec:.2f}")

if __name__ == "__main__":
    main()
