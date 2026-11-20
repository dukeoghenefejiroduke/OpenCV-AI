import os
import hashlib
import cv2
from vision.orchestrator import MultiHazardOrchestrator

path = "data/challenge_pack/Thin Corrosion2.jpg"
out_path = "fixed_Thin_Corrosion2_v5.jpg"

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

    fused_findings = report.get("authoritative_inspection_report", {}).get("regions", []) # or fused findings
    summary = report.get("inspection_summary", {})

    print(f"  - Total Findings: {report['total_findings_count']}")
    print(f"  - Summary: {summary}")
    print(f"  - Saved Annotated to: {out_path}")
    print(f"  - Saved Mask Overlay to: mask_Thin_Corrosion2.jpg\n")

    print("  - Fused Findings & Unique Region IDs (v5a / v5b):")
    # Access findings from detectors or orchestrator report
    # Let's inspect detectors['corrosion_detector']['findings'] or similar
    corr_res = report["detectors"]["corrosion_detector"]
    for f in corr_res.get("findings", []):
        print(f"    Region ID: {f.get('region_id')}, BBox: {f.get('bbox')}, Mask Pixels: {f.get('pixel_measurement', {}).get('mask_pixels')}, Pct ROI: {f.get('pixel_measurement', {}).get('estimated_percentage')}%")

print("\nThin Corrosion2 v5 verification complete.")
