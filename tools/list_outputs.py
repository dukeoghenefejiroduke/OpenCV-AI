"""Outputs Reporting Tool (tools/list_outputs.py).

Dry-run utility that reports file sizes and ages of items in data/outputs/ without deleting anything.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def list_outputs(output_dir: str = "data/outputs"):
    if not os.path.exists(output_dir):
        print(f"[Outputs] Directory {output_dir} does not exist.")
        return

    now = time.time()
    total_size = 0
    count = 0

    print(f"[Outputs Report] Scanning {output_dir} (Dry-run, never deletes):")
    print(f"{'Filename':<40} | {'Size (KB)':<10} | {'Age (Hours)'}")
    print("-" * 70)

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                st = os.stat(fpath)
                size_kb = st.st_size / 1024.0
                age_hours = (now - st.st_mtime) / 3600.0
                total_size += st.st_size
                count += 1
                rel_path = os.path.relpath(fpath, output_dir)
                print(f"{rel_path:<40} | {size_kb:<10.1f} | {age_hours:.1f}h")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    print("-" * 70)
    print(f"Total files: {count}, Total size: {total_size / (1024*1024.0):.2f} MB")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/outputs"
    list_outputs(output_dir=out_dir)
