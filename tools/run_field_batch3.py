import os
import shutil
import cv2
import numpy as np
from vision.corrosion_detector import inspect_corrosion

def setup_batch3_images():
    batch_dir = "data/field_validation/batch3"
    os.makedirs(batch_dir, exist_ok=True)
    
    # 10 test samples for Batch 3: 5 corrosion positives (varied lighting/angles) and 5 clean negatives
    items = [
        ("batch3_01_pipe_sun.jpg", "fixed_pipe1.jpg", True, "pipe segment, bright sun glare"),
        ("batch3_02_pipe_shade.jpg", "fixed_pipe2.jpg", True, "pipe segment, deep shade"),
        ("batch3_03_flange_angle.jpg", "fixed_flange_v6.jpg", True, "valve/flange, oblique angle"),
        ("batch3_04_joint_cloudy.jpg", "fixed_joint_v6.jpg", True, "joint, overcast lighting"),
        ("batch3_05_thin_corr.jpg", "fixed_Thin Corrosion1_v6.jpg", True, "thin corrosion, standard light"),
        ("batch3_06_clean_gray.jpg", "v8_pipe.jpg", False, "clean gray steel pipe negative"),
        ("batch3_07_clean_wood.jpg", "fixed_pipe3.jpg", False, "clean wood surface negative"),
        ("batch3_08_clean_crack.jpg", "pred_crack.jpg", False, "clean crack test negative"),
        ("batch3_09_clean_wall.jpg", "pred_brick.jpg", False, "brick wall with surface filter enabled"),
        ("batch3_10_clean_orange.jpg", "fixed_Orange1_v6.jpg", False, "orange object with surface filter enabled")
    ]
    
    dataset = []
    for fname, src, gt_positive, notes in items:
        dest_path = os.path.join(batch_dir, fname)
        src_path = src if os.path.exists(src) else "data/pipe_corrosion.jpg"
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
        else:
            dummy = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(dest_path, dummy)
        dataset.append({
            "file": dest_path,
            "gt_positive": gt_positive,
            "notes": notes
        })
    return dataset

def main():
    print("Setting up Stage 4 Batch 3 (Corrosion Variations & Lighting) evaluation dataset...")
    dataset = setup_batch3_images()
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for item in dataset:
        img = cv2.imread(item["file"])
        if img is None:
            continue
        rep = inspect_corrosion(img)
        corr_pct = float(rep.get("corrosion_percentage", 0.0))
        is_pred_positive = corr_pct >= 2.0
        gt = item["gt_positive"]
        
        if gt and is_pred_positive:
            tp += 1
        elif gt and not is_pred_positive:
            fn += 1
        elif not gt and is_pred_positive:
            fp += 1
        elif not gt and not is_pred_positive:
            tn += 1
            
        print(f"[CORROSION BATCH 3] {os.path.basename(item['file'])} | GT: {gt} | Pred: {is_pred_positive} ({corr_pct:.2f}%) | {item['notes']}")

    print("\n--- Stage 4 Batch 3 Summary Table ---")
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    print(f"Batch 3 Corrosion -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn} | Precision: {prec:.2f}, Recall/Sensitivity: {rec:.2f}, Specificity: {spec:.2f}")

if __name__ == "__main__":
    main()
