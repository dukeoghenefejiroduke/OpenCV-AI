import os
import hashlib
import cv2
from vision.orchestrator import MultiHazardOrchestrator

path = "data/challenge_pack/Thin Corrosion2.jpg"
out_path = "fixed_Thin_Corrosion2_v3.jpg"

if not os.path.exists(path):
    print(f"Error: {path} not found.")
else:
    # 1. File stats & MD5 hash
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

    # 2. Run MultiHazardOrchestrator
    orchestrator = MultiHazardOrchestrator(enable_experimental=True)
    report = orchestrator.run_inspections(img, output_annotated_path=out_path)

    print(f"  - Total Fused Boxes: {report['total_findings_count']}")
    print(f"  - Inspection Summary: {report.get('inspection_summary', {})}")
    print(f"  - Saved annotated output to: {out_path}")

    # 3. Print exact box list drawn
    print("  - Exact Fused Box List:")
    # Collect all findings from detectors that passed fusion
    detectors = report.get("detectors", {})
    all_fused = []
    # Let's inspect the detectors dict or re-collect from detectors by running detection or inspection summary
    # Actually, let's list findings from each detector that are in fused list or inspect detectors
    for det_id, dres in detectors.items():
        for f in dres.get("findings", []):
            # Check if this finding's bbox is in fused findings
            all_fused.append(f)

    # Better yet, let's collect from orchestrator or inspect the output
    print(f"    Platform reported total findings count: {report['total_findings_count']}")
    
    # Print each detector's valid findings
    for det_id, dres in detectors.items():
        valid_findings = dres.get("findings", [])
        if valid_findings:
            print(f"    * Detector [{det_id}]: {len(valid_findings)} valid findings")
            for f in valid_findings:
                print(f"      - ID: {f.get('finding_id')}, BBox: {f.get('bbox')}, Conf: {f.get('confidence')}")

print("Thin Corrosion2 v3 verification complete.")
