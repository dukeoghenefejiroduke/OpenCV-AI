import os
import cv2
import numpy as np
from glob import glob

from vision.corrosion_detector import CorrosionDetector
from vision.experimental_detector import (
    ExperimentalCrackDetector,
    ExperimentalCoatingDetector,
    ExperimentalLeakageDetector,
    ExperimentalInsulationDetector
)
from vision.orchestrator import MultiHazardOrchestrator

def main():
    output_dir = "data/debug_cv"
    os.makedirs(output_dir, exist_ok=True)

    detectors = [
        CorrosionDetector(),
        ExperimentalCrackDetector(),
        ExperimentalCoatingDetector(),
        ExperimentalLeakageDetector(),
        ExperimentalInsulationDetector()
    ]

    orchestrator = MultiHazardOrchestrator(enable_experimental=True)

    # Collect sample images
    image_paths = []
    image_paths.append("data/pipe_corrosion.jpg")
    image_paths.extend(glob("data/challenge_pack/*.jpg"))
    image_paths.extend(glob("*.jpg"))
    image_paths = [p for p in image_paths if os.path.exists(p)]
    # Deduplicate while preserving order
    image_paths = list(dict.fromkeys(image_paths))

    print(f"Found {len(image_paths)} sample images for verification and overlay generation.")

    detector_counts = {det.detector_id: 0 for det in detectors}
    total_detections = 0

    category_colors = {
        "corrosion": (36, 191, 251),    # Orange/Amber
        "crack": (246, 130, 59),        # Blue
        "coating": (247, 85, 168),      # Purple
        "leakage": (212, 182, 6),       # Cyan
        "insulation": (129, 185, 16),   # Green
    }

    for img_path in image_paths[:15]: # Process up to 15 key images to keep output concise
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        annotated = img.copy()
        base_name = os.path.basename(img_path)

        for det in detectors:
            res = det.detect(img)
            findings = res.get("findings", [])
            count = len(findings)
            detector_counts[det.detector_id] += count
            total_detections += count

            cat = det.defect_category
            color = category_colors.get(cat, (0, 0, 255))

            for f in findings:
                bbox = f.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    reg_id = f.get("region_id", "REG")
                    cv2.putText(annotated, f"{reg_id} [{cat[:3]}]", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        out_path = os.path.join(output_dir, f"overlay_{base_name}")
        cv2.imwrite(out_path, annotated)
        print(f"Processed {base_name} -> Saved overlay to {out_path}")

    print("\n--- PER-DETECTOR DETECTION COUNTS ---")
    for det_id, count in detector_counts.items():
        print(f"  {det_id}: {count} total detections")
    print(f"Total Detections Across All Detectors: {total_detections}")

if __name__ == "__main__":
    main()
