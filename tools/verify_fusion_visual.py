import os
import cv2
from vision.orchestrator import MultiHazardOrchestrator

test_images = [
    ("data/pipe_corrosion.jpg", "fixed_pipe1_v2.jpg"),
    ("data/challenge_pack/Thin Corrosion1.jpg", "fixed_pipe2.jpg"),
    ("data/challenge_pack/Orange1.jpg", "fixed_pipe3.jpg")
]

orchestrator = MultiHazardOrchestrator(enable_experimental=True)

for path, out_path in test_images:
    if not os.path.exists(path):
        path = "data/pipe_corrosion.jpg"
    
    img = cv2.imread(path)
    if img is None:
        continue

    report = orchestrator.run_inspections(img, output_annotated_path=out_path)
    summary = report.get("inspection_summary", {})

    print(f"Image: {path}")
    print(f"  - Total Fused Boxes: {report['total_findings_count']}")
    print(f"  - Summary Breakdown: {summary}")
    print(f"  - Saved to: {out_path}\n")

print("v2 verification complete.")
