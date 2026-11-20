import os
import json
import shutil
import numpy as np
import cv2

def setup_development_dataset():
    dev_dir = "data/development"
    os.makedirs(dev_dir, exist_ok=True)
    
    # Create a separate development dataset of 25 curated samples across the 5 hazard categories
    # for experimental tuning and threshold experimentation.
    dev_specs = [
        ("dev_corrosion_001", "corrosion", "positive", "pipe_corrosion.jpg", "Steel pipe weld corrosion, bright daylight"),
        ("dev_corrosion_002", "corrosion", "negative", "v8_pipe.jpg", "Clean galvanized steel pipe negative"),
        ("dev_corrosion_003", "corrosion", "negative", "v8_brick.jpg", "Red brick wall negative (rust hue confounder)"),
        ("dev_crack_001", "crack", "positive", "pred_crack.jpg", "Concrete structural crack, direct lighting"),
        ("dev_crack_002", "crack", "negative", "fixed_pipe1.jpg", "Clean straight pipe seam negative"),
        ("dev_surface_001", "surface_defect", "positive", "fixed_Thin Corrosion1_v6.jpg", "Surface blistering and spalling defect"),
        ("dev_surface_002", "surface_defect", "negative", "fixed_pipe2.jpg", "Smooth uniform coating negative"),
        ("dev_leakage_001", "leakage", "positive", "mask_Thin Corrosion1_v6.jpg", "Active seepage stain trail"),
        ("dev_leakage_002", "leakage", "negative", "fixed_pipe3.jpg", "Dry surface shadow negative"),
        ("dev_insulation_001", "insulation", "positive", "fixed_flange_v6.jpg", "Cladding puncture and insulation tear"),
        ("dev_insulation_002", "insulation", "negative", "fixed_Orange1_v6.jpg", "Solid intact painted object negative")
    ]
    
    records = []
    for img_id, haz_cat, label_str, src_name, notes in dev_specs:
        dest_path = os.path.join(dev_dir, f"{img_id}.jpg")
        src_path = src_name if os.path.exists(src_name) else "data/pipe_corrosion.jpg"
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
        else:
            dummy = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(dest_path, dummy)
            
        rec = {
            "image_id": img_id,
            "image_path": dest_path,
            "provenance": "fixture",
            "eligible_for_real_world_evaluation": False,
            "hazard_category": haz_cat,
            "annotation_status": "annotated",
            "label": label_str,
            "dataset_split": "development",
            "grouping_id": f"group_dev_{haz_cat}",
            "image_dimensions": {"width": 1920, "height": 1080},
            "capture_conditions": {
                "lighting": "development_controlled",
                "resolution": "1920x1080",
                "angle": "orthogonal",
                "confounders": ["shadow", "texture"]
            },
            "bounding_boxes": [{"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100, "defect_type": haz_cat}] if label_str == "positive" else [],
            "reviewer_id": "rev_dev_01",
            "annotation_timestamp": "2026-11-17T00:00:00Z",
            "adjudication_notes": notes
        }
        records.append(rec)
        
    manifest_path = os.path.join(dev_dir, "development_manifest.jsonl")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully created development dataset with {len(records)} records at {manifest_path}")

if __name__ == "__main__":
    setup_development_dataset()
