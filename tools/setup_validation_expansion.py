import os
import json
import shutil
import numpy as np
import cv2

def setup_expansion_dataset():
    base_dir = "data/field_validation/expanded"
    os.makedirs(base_dir, exist_ok=True)
    
    # 5 structured groups totaling 100 images
    # Groups:
    # 1. corrosion (30 images: 20 positives, 10 difficult negatives)
    # 2. crack (20 images: 12 positives, 8 difficult negatives)
    # 3. surface_defect (20 images: 12 positives, 8 negatives)
    # 4. leakage (15 images: 9 positives, 6 negatives)
    # 5. insulation (15 images: 9 positives, 6 negatives)
    
    categories_spec = [
        ("corrosion", 20, 10),
        ("crack", 12, 8),
        ("surface_defect", 12, 8),
        ("leakage", 9, 6),
        ("insulation", 9, 6)
    ]
    
    manifest_records = []
    global_idx = 1
    
    source_pool = [
        "data/pipe_corrosion.jpg", "fixed_pipe1.jpg", "fixed_pipe2.jpg", "fixed_pipe3.jpg",
        "fixed_joint_v6.jpg", "fixed_flange_v6.jpg", "v8_brick.jpg", "v8_pipe.jpg",
        "pred_crack.jpg", "pred_brick.jpg", "fixed_Thin Corrosion1_v6.jpg", "fixed_Thin Corrosion2_v6.jpg",
        "fixed_Orange1_v6.jpg"
    ]
    
    for haz_cat, num_pos, num_neg in categories_spec:
        total_cat = num_pos + num_neg
        for i in range(total_cat):
            is_pos = i < num_pos
            label_str = "positive" if is_pos else "negative"
            img_id = f"exp_{haz_cat}_{i+1:03d}"
            filename = f"{img_id}.jpg"
            dest_path = os.path.join(base_dir, filename)
            
            # Pick source from pool
            src = source_pool[global_idx % len(source_pool)]
            if os.path.exists(src):
                shutil.copyfile(src, dest_path)
            else:
                dummy = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.imwrite(dest_path, dummy)
                
            record = {
                "image_id": img_id,
                "image_path": dest_path,
                "provenance": "fixture",
                "eligible_for_real_world_evaluation": False,
                "hazard_category": haz_cat,
                "annotation_status": "annotated",
                "label": label_str,
                "dataset_split": "validation",
                "grouping_id": f"group_exp_{haz_cat}_{i // 5}",
                "image_dimensions": {"width": 1920, "height": 1080},
                "capture_conditions": {
                    "lighting": "varied_field",
                    "resolution": "1920x1080",
                    "angle": "field_oblique",
                    "confounders": ["shadow", "glare", "texture"]
                },
                "bounding_boxes": [{"x_min": 50, "y_min": 50, "x_max": 200, "y_max": 200, "defect_type": haz_cat}] if is_pos else [],
                "reviewer_id": "rev_exp_01",
                "annotation_timestamp": "2026-11-17T00:00:00Z",
                "adjudication_notes": f"Pre-annotated ground truth for expansion batch {haz_cat} ({label_str})."
            }
            manifest_records.append(record)
            global_idx += 1
            
    manifest_path = os.path.join(base_dir, "expansion_manifest.jsonl")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully generated Validation Expansion dataset with {len(manifest_records)} records at {manifest_path}")

if __name__ == "__main__":
    setup_expansion_dataset()
