import os
import cv2
import json
from vision.orchestrator import MultiHazardOrchestrator

def run_verification():
    image_path = "data/pipe_corrosion.jpg"
    assert os.path.exists(image_path), f"Sample image {image_path} not found."
    
    img = cv2.imread(image_path)
    os.makedirs("data/debug_cv", exist_ok=True)
    out_annotated = "data/debug_cv/live_api_check.jpg"
    out_roi = "data/debug_cv/foreground_roi.jpg"
    
    orch = MultiHazardOrchestrator(enable_experimental=True)
    result = orch.run_inspections(img, output_annotated_path=out_annotated, output_roi_path=out_roi, use_roi=True)
    
    print("Multi-Hazard Inspection Result Summary:")
    print(json.dumps(result.get("inspection_summary", {}), indent=2))
    
    detectors = result.get("detectors", {})
    all_findings = []
    for det_id, det_res in detectors.items():
        findings = det_res.get("findings", [])
        print(f"Detector {det_id}: {len(findings)} findings")
        for f in findings:
            all_findings.append({
                "id": f.get("region_id"),
                "type": f.get("defect_category"),
                "bbox": f.get("bbox"),
                "confidence": f.get("confidence")
            })
            
    print("\nExact JSON List of Returned Detections:")
    print(json.dumps(all_findings, indent=2))
    print(f"\nSaved verification image to {out_annotated}")

if __name__ == "__main__":
    run_verification()
