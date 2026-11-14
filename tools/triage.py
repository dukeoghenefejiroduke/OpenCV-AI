"""CementSight Failure Triage Tool (tools/triage.py).

Runs the multi-hazard inspection pipeline across all images in specified target folders/files,
saving per-detector findings and annotated overlay PNGs into a NEW folder under data/triage/,
writing timestamped summary files (never a fixed name), and exiting with code 2 if zero images are found.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import json
import hashlib
import time
import cv2
import numpy as np

from vision.orchestrator import MultiHazardOrchestrator
from vision.quality_gate import ImageQualityGate


def run_triage(target_paths: list, feedback_dir: str = "data/field_feedback", output_dir: str = "data/triage") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    orchestrator = MultiHazardOrchestrator(enable_experimental_buildup=True)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []

    for p in target_paths:
        if os.path.isfile(p):
            image_paths.append(p)
        elif os.path.isdir(p):
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))

    # Deduplicate while preserving order
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"[Triage Error] No valid image files found in targets: {target_paths}", file=sys.stderr)
        sys.exit(2)

    summary = {
        "total_images": len(image_paths),
        "processed": 0,
        "failed": 0,
        "zero_findings": [],
        "many_findings": [],
        "quality_gate_verdicts": {},
        "results": []
    }

    print(f"[Triage] Scanning {len(image_paths)} images across targets: {target_paths}")

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                summary["failed"] += 1
                continue

            report = orchestrator.run_inspections(img)
            summary["processed"] += 1

            q_verdict = report.get("quality_gate", {}).get("verdict", "UNKNOWN")
            summary["quality_gate_verdicts"][img_path] = q_verdict

            findings = report.get("all_findings", [])
            finding_count = len(findings)

            if finding_count == 0:
                summary["zero_findings"].append(img_path)
            elif finding_count >= 3:
                summary["many_findings"].append((img_path, finding_count))

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            sha = hashlib.sha256(img_path.encode()).hexdigest()[:10]
            sub_folder = os.path.join(output_dir, f"triage_{base_name}_{sha}")
            os.makedirs(sub_folder, exist_ok=True)

            report_path = os.path.join(sub_folder, "report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            overlay_img = img.copy()
            for finding in findings:
                bbox = finding.get("bounding_box")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.rectangle(overlay_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
                    cv2.putText(overlay_img, finding.get("detector_id", "defect"), (int(x), max(15, int(y)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            overlay_path = os.path.join(sub_folder, "annotated_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_img)

            summary["results"].append({
                "image_path": img_path,
                "quality_verdict": q_verdict,
                "finding_count": finding_count,
                "output_dir": sub_folder
            })
        except Exception as e:
            print(f"[Triage Error] Failed processing {img_path}: {e}")
            summary["failed"] += 1

    feedback_results = []
    if os.path.exists(feedback_dir):
        for fb_file in glob.glob(os.path.join(feedback_dir, "*.jsonl")):
            with open(fb_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            feedback_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"triage_summary_{timestamp_str}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[Triage] Completed. Processed: {summary['processed']}, Failed: {summary['failed']}")
    print(f"[Triage] Zero findings count: {len(summary['zero_findings'])}")
    print(f"[Triage] Many findings count: {len(summary['many_findings'])}")
    print(f"[Triage] Timestamped summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="CementSight Failure Triage Tool")
    parser.add_argument("paths", nargs="*", default=["data"], help="Target image files or directories to triage")
    parser.add_argument("--feedback-dir", default="data/field_feedback", help="Path to field feedback directory")
    parser.add_argument("--output-dir", default="data/triage", help="Path to output triage directory")
    args = parser.parse_args()

    run_triage(target_paths=args.paths, feedback_dir=args.feedback_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
