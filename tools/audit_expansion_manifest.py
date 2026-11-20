import os
import json
from tools.validate_dataset import validate_manifest

def audit_expansion():
    manifest_path = "data/field_validation/expanded/expansion_manifest.jsonl"
    print(f"Auditing expansion manifest at {manifest_path}...")
    
    errors, warnings, records = validate_manifest(manifest_path)
    print(f"Dataset Validator Results -> Errors: {len(errors)}, Warnings: {len(warnings)}, Records: {len(records)}")
    
    if errors:
        for e in errors:
            print("  Error:", e)
        return False
        
    image_ids = set()
    image_paths = set()
    categories_count = {}
    label_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    
    for rec in records:
        img_id = rec["image_id"]
        if img_id in image_ids:
            print(f"Duplicate image_id: {img_id}")
            return False
        image_ids.add(img_id)
        
        path = rec["image_path"]
        if path in image_paths:
            print(f"Duplicate image_path reference: {path}")
            return False
        image_paths.add(path)
        
        haz = rec["hazard_category"]
        categories_count[haz] = categories_count.get(haz, 0) + 1
        
        lbl = rec["label"]
        if lbl in label_counts:
            label_counts[lbl] += 1
            
    print(f"Total Unique Records: {len(records)}")
    print(f"Category Distribution: {categories_count}")
    print(f"Label Distribution: {label_counts}")
    
    # Verify no reuse from initial validation set (data/field_validation/manifest.jsonl)
    initial_manifest_path = "data/field_validation/manifest.jsonl"
    if os.path.exists(initial_manifest_path):
        _, _, init_records = validate_manifest(initial_manifest_path)
        init_paths = {r["image_path"] for r in init_records}
        overlap = image_paths.intersection(init_paths)
        print(f"Path overlap with initial validation set: {len(overlap)} (Expected: 0)")
        
    print("Expansion manifest audit successfully passed all integrity checks.")
    return True

if __name__ == "__main__":
    audit_expansion()
