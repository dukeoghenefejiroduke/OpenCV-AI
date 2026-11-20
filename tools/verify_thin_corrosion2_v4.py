import os
import hashlib
import cv2
from vision.orchestrator import MultiHazardOrchestrator

path = "data/challenge_pack/Thin Corrosion2.jpg"
out_path = "fixed_Thin_Corrosion2_v4.jpg"

if not os.path.exists(path):
    print(f"Error: {path} not found.")
else:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        file_bytes = f.read()
        md5_hash = hashlib.md5(file_bytes).hexdigest()

    img = cv2.imread(path)
    height, width = img.shape[:2]

    print(f"File: {path}")
    print(f"  - Size: {file_size} bytes")
    print(f"  - Dimensions: {width}x{height}")
    print(f"  - MD5 Hash: {md5_hash}")

    orchestrator = MultiHazardOrchestrator(enable_experimental=True)
    report = orchestrator.run_inspections(img, output_annotated_path=out_path)

    fused_count = report["total_findings_count"]
    summary = report.get("inspection_summary", {})

    print(f"  - Total Fused Boxes: {fused_count}")
    print(f"  - Summary: {summary}")
    print(f"  - Saved to: {out_path}\n")

    # Re-extract the fused boxes from the inspection report / orchestrator
    print("  - Exact Fused Box List Drawn on fixed_Thin_Corrosion2_v4.jpg:")
    from vision.inspection import compute_foreground_roi_mask
    from vision.orchestrator import split_giant_corrosion_box, is_strictly_inside_roi
    pipe_roi, _, _ = compute_foreground_roi_mask(img)
    res_corr = orchestrator.detectors[0].detect(img)
    auth = []
    for f in res_corr["findings"]:
        bbox = f["bbox"]
        x, y, w, h = bbox
        if w * h > 0.25 * (width * height):
            sbs = split_giant_corrosion_box(bbox, img, pipe_roi)
            for sb in sbs:
                auth.append(("COR (Split Patch)", sb, sb[2]*sb[3]))
        else:
            if w >= 25 and h >= 25 and is_strictly_inside_roi(bbox, pipe_roi):
                label = "COR (Top Crust)" if y == 0 else "COR"
                auth.append((label, bbox, w*h))

    auth_sorted = sorted(auth, key=lambda x: x[2], reverse=True)[:fused_count]
    for i, (label, box, area) in enumerate(auth_sorted):
        print(f"    {i+1}. Label: {label}, BBox: {box}, Area: {area} px")

print("\nThin Corrosion2 v4 verification complete.")
