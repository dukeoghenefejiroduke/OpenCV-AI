#!/usr/bin/env python3
"""Challenge Pack Evaluation Tool (tools/challenge_pack.py).

Reads data/challenge_pack/truth.csv (file,category,truth), runs default detectors
and experimental rust detector, prints per-category and overall results with Wilson intervals,
refuses if a listed file is missing, records detector errors explicitly, exits non-zero on error,
and writes timestamped outputs.
"""

import os
import sys
import csv
import json
import time
import math
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.inspection import inspect_corrosion
from vision.rust_detector import RustDetector


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denominator
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main():
    truth_csv = "data/challenge_pack/truth.csv"
    if not os.path.exists(truth_csv):
        print(f"Error: Challenge pack truth CSV not found at {truth_csv}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(truth_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            filepath = r.get("file", "").strip()
            if not os.path.exists(filepath):
                print(f"Error: Challenge pack listed file missing: {filepath}", file=sys.stderr)
                sys.exit(1)
            rows.append(r)

    print(f"Loaded {len(rows)} challenge items from {truth_csv}.")

    results = []
    rust_detector = RustDetector()
    detector_errors = []

    for r in rows:
        fpath = r["file"]
        cat = r["category"]
        truth = int(r["truth"])

        img = cv2.imread(fpath)
        if img is None:
            print(f"Error: Failed to read image {fpath}", file=sys.stderr)
            sys.exit(1)

        # Default detector report
        default_rep = inspect_corrosion(img)
        default_corr = default_rep.get("corrosion_percentage", 0.0)
        default_flagged = default_corr > 2.0

        # Experimental rust detector report with robust error handling
        try:
            rust_res = rust_detector.detect(img) if hasattr(rust_detector, "detect") else {}
            if not isinstance(rust_res, dict):
                raise ValueError("RustDetector.detect did not return a dictionary")
            
            if rust_res.get("errors"):
                err_msg = ", ".join(str(e) for e in rust_res["errors"])
                rust_score = f"error: {err_msg}"
                rust_flagged = False
                detector_errors.append((fpath, err_msg))
            elif rust_res.get("execution_status") in ("detector_failed", "STATUS_DETECTOR_FAILED"):
                err_msg = "detector execution failed"
                rust_score = f"error: {err_msg}"
                rust_flagged = False
                detector_errors.append((fpath, err_msg))
            else:
                rust_meta = rust_res.get("metadata", {})
                rust_score = float(rust_meta.get("rust_percentage", 0.0))
                rust_flagged = rust_score > 2.0
        except Exception as e:
            err_msg = str(e)
            rust_score = f"error: {err_msg}"
            rust_flagged = False
            detector_errors.append((fpath, err_msg))

        results.append({
            "file": fpath,
            "category": cat,
            "truth": truth,
            "default_corrosion_pct": default_corr,
            "default_flagged": default_flagged,
            "rust_detector_pct": rust_score,
            "rust_flagged": rust_flagged,
        })

    if detector_errors:
        print(f"Error: Encountered {len(detector_errors)} detector errors during challenge evaluation.", file=sys.stderr)
        for fpath, emsg in detector_errors:
            print(f"  {fpath}: {emsg}", file=sys.stderr)
        sys.exit(1)

    # Summarize per category
    categories = set(r["category"] for r in rows)
    summary = {}
    total_default_succ = 0
    total_rust_succ = 0
    total_count = len(rows)

    for cat in categories:
        cat_rows = [res for res in results if res["category"] == cat]
        c_total = len(cat_rows)
        c_def_corr = sum(1 for res in cat_rows if (res["default_flagged"] == bool(res["truth"])))
        c_rust_corr = sum(1 for res in cat_rows if (res["rust_flagged"] == bool(res["truth"])))
        total_default_succ += c_def_corr
        total_rust_succ += c_rust_corr
        summary[cat] = {
            "total": c_total,
            "default_correct": c_def_corr,
            "rust_correct": c_rust_corr,
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/challenge_report_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": results, "summary": summary}, f, indent=2)

    print("\n=== CHALLENGE PACK EVALUATION RESULTS ===")
    for cat, data in summary.items():
        print(f"Category: {cat} (n={data['total']})")
        dp, d_low, d_high = wilson_interval(data["default_correct"], data["total"])
        rp, r_low, r_high = wilson_interval(data["rust_correct"], data["total"])
        print(f"  Default Detector Accuracy: {dp:.2f} [Wilson 95% CI: {d_low:.2f} - {d_high:.2f}]")
        print(f"  Rust Detector Accuracy   : {rp:.2f} [Wilson 95% CI: {r_low:.2f} - {r_high:.2f}]")

    print(f"Detailed challenge report saved to: {output_path}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
