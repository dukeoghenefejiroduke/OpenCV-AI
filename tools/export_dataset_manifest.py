#!/usr/bin/env python3
"""Export SHA-256 Dataset Manifest (tools/export_dataset_manifest.py).

Generates a JSON manifest with SHA-256 checksums for files in a dataset directory.
"""

import os
import sys
import json
import hashlib

def generate_manifest(data_dir: str = "data/field_validation", output_path: str = "data/outputs/dataset_manifest.json"):
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            fpath = os.path.join(root, file)
            if os.path.isfile(fpath):
                hasher = hashlib.sha256()
                with open(fpath, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                rel_path = os.path.relpath(fpath, ".")
                manifest[rel_path] = hasher.hexdigest()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported manifest with {len(manifest)} entries to {output_path}")

if __name__ == "__main__":
    dir_arg = sys.argv[1] if len(sys.argv) > 1 else "data/field_validation"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "data/outputs/dataset_manifest.json"
    generate_manifest(dir_arg, out_arg)
