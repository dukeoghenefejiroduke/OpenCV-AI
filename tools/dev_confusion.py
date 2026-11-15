"""Development Confusion & Comparative Evaluation Tool (tools/dev_confusion.py).

Takes --photos DIR and --truth CSV (columns file,truth).
Processes ONLY files listed in the CSV from that folder; errors out if unlisted files exist or listed files are missing.
Prints per-photo details (truth, quality verdict and reasons, old corrosion %, new rust %, flagged status)
and summary statistics (rust photos flagged out of 8, clean photos flagged out of 4, per detector).
Saves output to a timestamped JSON file in data/outputs/.
"""

import os
import sys
import csv
import json
import glob
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.inspection import inspect_corrosion
from vision.rust_detector import RustDetector
from vision.quality_gate import ImageQualityGate


def run_confusion(photos_dir: str, truth_csv: str):
    if not os.path.isdir(photos_dir):
        print(f"[Error] Photos directory does not exist: {photos_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(truth_csv):
        print(f"[Error] Truth CSV file does not exist: {truth_csv}", file=sys.stderr)
        sys.exit(1)

    # 1. Load truth CSV
    truth_map = {}
    with open(truth_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("file", "").strip()
            tval = row.get("truth", "").strip()
            if fname:
                truth_map[fname] = tval

    # 2. Get image files in photos_dir
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    disk_files = []
    for ext in extensions:
        for p in glob.glob(os.path.join(photos_dir, "**", ext), recursive=True):
            disk_files.append(os.path.basename(p))
    disk_files = set(disk_files)

    expected_files = set(truth_map.keys())

    # Check for unlisted files in directory
    unlisted = disk_files - expected_files
    if unlisted:
        print(f"[Error] Refusing input folder: found unlisted files not in truth CSV: {unlisted}", file=sys.stderr)
        sys.exit(2)

    # Check for missing files
    missing = expected_files - disk_files
    if missing:
        print(f"[Error] Listed truth files missing from directory {photos_dir}: {missing}", file=sys.stderr)
        sys.exit(3)

    quality_gate = ImageQualityGate()
    rust_det = RustDetector()

    results = []
    rust_total = 0
    rust_old_flagged = 0
    rust_new_flagged = 0

    clean_total = 0
    clean_old_flagged = 0
    clean_new_flagged = 0

    print("\n" + "=" * 160)
    print(f"{'PHOTO':<18} | {'TRUTH':<18} | {'QUALITY VERDICT (REASONS)':<60} | {'OLD CORR %':<11} | {'NEW RUST %':<11} | {'OLD FLAG':<8} | {'NEW FLAG':<8}")
    print("=" * 160)

    for filename in sorted(truth_map.keys()):
        truth = truth_map[filename]
        img_path = os.path.join(photos_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Error] Could not load image: {img_path}", file=sys.stderr)
            sys.exit(4)

        # Quality gate
        q_res = quality_gate.evaluate(img)
        q_verdict = q_res.get("verdict", "UNKNOWN")
        q_reasons = "; ".join(q_res.get("reasons", [])) or "None"

        # Old corrosion detector
        try:
            old_res = inspect_corrosion(img, use_roi=True)
            old_pct = float(old_res.get("corrosion_percentage", 0.0))
        except Exception:
            old_pct = 0.0

        # New rust detector
        try:
            new_res = rust_det.detect(img)
            new_pct = float(new_res.get("metadata", {}).get("rust_percentage", 0.0))
        except Exception:
            new_pct = 0.0

        old_flagged = old_pct > 0.0
        new_flagged = new_pct > 0.0

        is_rust_class = (truth == "rust" or truth == "rust_dim_light")
        if is_rust_class:
            rust_total += 1
            if old_flagged: rust_old_flagged += 1
            if new_flagged: rust_new_flagged += 1
        else:
            clean_total += 1
            if old_flagged: clean_old_flagged += 1
            if new_flagged: clean_new_flagged += 1

        quality_info = f"{q_verdict}" if not q_res.get("reasons") else f"{q_verdict}: {q_reasons}"
        print(f"{filename:<18} | {truth:<18} | {quality_info:<60} | {old_pct:<10.2f}% | {new_pct:<10.2f}% | {'YES' if old_flagged else 'NO':<8} | {'YES' if new_flagged else 'NO':<8}")

        results.append({
            "file": filename,
            "truth": truth,
            "quality_verdict": q_verdict,
            "quality_reasons": q_res.get("reasons", []),
            "old_corrosion_percentage": old_pct,
            "new_rust_percentage": new_pct,
            "old_flagged": old_flagged,
            "new_flagged": new_flagged
        })

    print("=" * 160)
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total Rust Photos (truth == rust or rust_dim_light): {rust_total}")
    print(f"  - Old Corrosion Detector Flagged: {rust_old_flagged} / {rust_total}")
    print(f"  - New Rust Detector Flagged     : {rust_new_flagged} / {rust_total}")
    print(f"Total Clean Photos (truth == clean)                  : {clean_total}")
    print(f"  - Old Corrosion Detector Flagged (False Alarms)    : {clean_old_flagged} / {clean_total}")
    print(f"  - New Rust Detector Flagged (False Alarms)         : {clean_new_flagged} / {clean_total}")
    print("--------------------------\n")

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/outputs", exist_ok=True)
    out_path = os.path.join("data/outputs", f"dev_confusion_summary_{timestamp_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp_str,
            "photos_dir": photos_dir,
            "truth_csv": truth_csv,
            "summary": {
                "rust_total": rust_total,
                "rust_old_flagged": rust_old_flagged,
                "rust_new_flagged": rust_new_flagged,
                "clean_total": clean_total,
                "clean_old_flagged": clean_old_flagged,
                "clean_new_flagged": clean_new_flagged
            },
            "results": results
        }, f, indent=2)

    print(f"[Dev Confusion] Summary report saved to {out_path}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CementSight Dev Confusion & Comparative Evaluation Tool")
    parser.add_argument("--photos", required=True, help="Path to photos directory")
    parser.add_argument("--truth", required=True, help="Path to truth CSV file")
    args = parser.parse_args()

    run_confusion(photos_dir=args.photos, truth_csv=args.truth)


if __name__ == "__main__":
    main()
