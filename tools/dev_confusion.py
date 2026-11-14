"""Development Confusion & Comparative Evaluation Tool (tools/dev_confusion.py).

Runs 12 development photos (or target image directory) with ground truth labels,
evaluating image quality gate, old corrosion inspector vs new rust detector,
and prints a per-photo comparative table.
"""

import os
import sys
import json
import glob
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.inspection import inspect_corrosion
from vision.rust_detector import RustDetector
from vision.quality_gate import ImageQualityGate


def run_dev_confusion(image_dir: str = "data"):
    quality_gate = ImageQualityGate()
    rust_det = RustDetector()

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    image_paths = sorted(list(set(image_paths)))[:12]

    if not image_paths:
        print(f"[Dev Confusion] No images found in {image_dir}")
        return

    print("\n" + "=" * 105)
    print(f"{'PHOTO / IMAGE':<35} | {'TRUTH (DEV)':<18} | {'QUALITY VERDICT':<15} | {'OLD CORR %':<12} | {'NEW RUST %':<12}")
    print("=" * 105)

    results = []
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        name_lower = basename.lower()
        if any(k in name_lower for k in ["rust2", "joint2", "cluster", "rust", "joint"]):
            truth = "rust"
        elif any(k in name_lower for k in ["dark", "dark2"]):
            truth = "rust in dim light"
        else:
            truth = "clean"

        # 1. Quality gate
        q_res = quality_gate.evaluate(img)
        q_verdict = q_res.get("verdict", "UNKNOWN")

        # 2. Old corrosion inspector
        try:
            old_res = inspect_corrosion(img, use_roi=True)
            old_pct = old_res.get("corrosion_percentage", 0.0)
        except Exception:
            old_pct = 0.0

        # 3. New rust detector
        try:
            new_res = rust_det.detect(img)
            new_pct = new_res.get("metadata", {}).get("rust_percentage", 0.0)
        except Exception:
            new_pct = 0.0

        print(f"{basename:<35} | {truth:<18} | {q_verdict:<15} | {old_pct:<11.2f}% | {new_pct:<11.2f}%")

        results.append({
            "image": basename,
            "truth": truth,
            "quality_verdict": q_verdict,
            "old_corrosion_percentage": old_pct,
            "new_rust_percentage": new_pct
        })

    print("=" * 105)
    os.makedirs("data/outputs", exist_ok=True)
    report_path = "data/outputs/dev_confusion_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Dev Confusion] Report saved to {report_path}\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_dev_confusion(target)
