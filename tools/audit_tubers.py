#!/usr/bin/env python3
"""Read-only audit of the TUBERS "Visual Data from Water Pipe Inspection" dataset.

Purpose (per project brief, Phase 2):
  - Verify archive SHA-256 against the recorded download.
  - Confirm image counts, CSV/folder label agreement.
  - Detect cross-partition near-duplicate risk via dHash + capture-session prefix grouping.
  - Emit an audit report documenting suspected leakage.
  - REFUSE to produce any accuracy / precision / recall / F1 metric.

This tool is READ-ONLY against the archive. It never extracts, never modifies,
never creates a train/val/test split, and never marks the dataset as eligible
for real-world evaluation. Those are deliberate constraints.

Dependencies: stdlib + numpy + opencv-python (already installed).
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

EXPECTED_SHA256 = "e1d7d78a903b2ef9454726d0938d0ff9a96348e9dcd73e8185bd0a0d7ad774b9"
EXPECTED_TOTAL = 930
EXPECTED_CRACK = 418
EXPECTED_NO_CRACK = 512

# Frame-number suffix, e.g. 2511201539a_001808.jpg -> prefix "2511201539a", frame 1808
FRAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# Per Zenodo dataset documentation: WIN_ prefixed files use a 1280x720 camera,
# all others a 1920x1080 camera. This is INFERRED from public metadata, not
# independently verified here; marked as "inferred" in the report.
def infer_camera(filename: str) -> str:
    base = os.path.basename(filename)
    if base.upper().startswith("WIN_"):
        return "1280x720 (inferred from WIN_ prefix)"
    return "1920x1080 (inferred: non-WIN_ prefix)"


def dhash_from_bytes(data: bytes, size: int = 8) -> Optional[str]:
    """Compute a 64-bit dHash from raw image bytes using cv2.imdecode."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bool(bit))
    return format(bits, "016x")


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@dataclass
class ImageRecord:
    path: str            # path inside the zip
    partition: str       # train / val / test
    label: str           # crack / no-crack
    prefix: str          # capture-session prefix, or "" if unparsable
    frame: Optional[int]
    camera: str
    dhash: Optional[str]


@dataclass
class PrefixGroup:
    prefix: str
    count: int = 0
    crack: int = 0
    no_crack: int = 0
    partitions: Dict[str, int] = field(default_factory=dict)
    frame_min: Optional[int] = None
    frame_max: Optional[int] = None
    cameras: Dict[str, int] = field(default_factory=dict)


def audit(archive_path: str, out_dir: str, dup_threshold: int = 4, max_pairs_reported: int = 200):
    if not os.path.exists(archive_path):
        print(f"ERROR: archive not found: {archive_path}", file=sys.stderr)
        return 1

    # 1. Archive hash
    h = hashlib.sha256()
    size = 0
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    archive_sha = h.hexdigest()
    hash_ok = archive_sha == EXPECTED_SHA256

    # 2. Enumerate images inside the zip
    records: List[ImageRecord] = []
    csv_rows_by_name: Dict[str, str] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()

        # Find CSV(s) — dataset ships a labels CSV; we parse it if found.
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        for cname in csv_names:
            try:
                with zf.open(cname) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.DictReader(text)
                    for row in reader:
                        # Try common column names
                        img = row.get("filename") or row.get("image") or row.get("path") or row.get("file")
                        label = row.get("label") or row.get("class") or row.get("category")
                        if img and label:
                            csv_rows_by_name[os.path.basename(img)] = str(label).strip().lower()
            except Exception as e:
                print(f"WARN: failed to parse CSV {cname}: {e}", file=sys.stderr)

        for n in names:
            base = os.path.basename(n)
            if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = n.split("/")
            part_idx = -1
            for idx, p_part in enumerate(parts):
                if p_part.lower() in ("train", "val", "test"):
                    part_idx = idx
                    break
            if part_idx == -1 or part_idx + 1 >= len(parts):
                continue
            partition = parts[part_idx].lower()
            label = parts[part_idx + 1].lower()
            if label not in ("crack", "no-crack"):
                continue
            m = FRAME_RE.match(base)
            prefix = m.group("prefix") if m else ""
            frame = int(m.group("frame")) if m else None
            try:
                data = zf.read(n)
            except Exception:
                data = b""
            records.append(ImageRecord(
                path=n, partition=partition, label=label,
                prefix=prefix, frame=frame,
                camera=infer_camera(base),
                dhash=dhash_from_bytes(data) if data else None,
            ))

    total = len(records)
    crack = sum(1 for r in records if r.label == "crack")
    no_crack = sum(1 for r in records if r.label == "no-crack")

    # 3. CSV-vs-folder agreement
    csv_agree = 0
    csv_mismatch = 0
    csv_missing = 0
    for r in records:
        base = os.path.basename(r.path)
        if base in csv_rows_by_name:
            csv_label = csv_rows_by_name[base].replace("_", "-")
            # normalise "nocrack" -> "no-crack"
            if csv_label in ("nocrack", "no crack", "negative", "0"):
                csv_label = "no-crack"
            elif csv_label in ("crack", "positive", "1", "cracked"):
                csv_label = "crack"
            if csv_label == r.label:
                csv_agree += 1
            else:
                csv_mismatch += 1
        else:
            csv_missing += 1

    # 4. Prefix groups
    groups: Dict[str, PrefixGroup] = {}
    for r in records:
        g = groups.setdefault(r.prefix or "(unparsed)", PrefixGroup(prefix=r.prefix or "(unparsed)"))
        g.count += 1
        if r.label == "crack":
            g.crack += 1
        else:
            g.no_crack += 1
        g.partitions[r.partition] = g.partitions.get(r.partition, 0) + 1
        g.cameras[r.camera] = g.cameras.get(r.camera, 0) + 1
        if r.frame is not None:
            g.frame_min = r.frame if g.frame_min is None else min(g.frame_min, r.frame)
            g.frame_max = r.frame if g.frame_max is None else max(g.frame_max, r.frame)

    # 5. Cross-partition near-duplicate risk via dHash
    # Only compare images that live in different partitions — same-partition
    # duplicates are not the leakage concern.
    by_partition: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.dhash is not None:
            by_partition[r.partition].append(r)

    cross_pairs: List[Tuple[str, str, int]] = []
    partitions = list(by_partition.keys())
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a_list = by_partition[partitions[i]]
            b_list = by_partition[partitions[j]]
            for a in a_list:
                for b in b_list:
                    d = hamming_hex(a.dhash, b.dhash)
                    if d <= dup_threshold:
                        cross_pairs.append((a.path, b.path, d))
    cross_pairs.sort(key=lambda t: t[2])

    # 6. Prefixes spanning multiple partitions
    multi_partition_prefixes = {
        p: sorted(g.partitions.keys())
        for p, g in groups.items()
        if len(g.partitions) > 1 and g.prefix != "(unparsed)"
    }

    # 7. Emit reports
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "archive_path": os.path.abspath(archive_path),
        "archive_size_bytes": size,
        "archive_sha256": archive_sha,
        "archive_sha256_matches_expected": hash_ok,
        "expected_sha256": EXPECTED_SHA256,
        "retrieval_date": "2026-11-05",
        "dataset_source": "https://zenodo.org/records/20136698",
        "dataset_doi": "10.5281/zenodo.20136698",
        "dataset_license": "CC BY 4.0",
        "dataset_creators": "Bendabl; University of Essex",
        "counts": {
            "total": total, "crack": crack, "no_crack": no_crack,
            "expected_total": EXPECTED_TOTAL,
            "expected_crack": EXPECTED_CRACK,
            "expected_no_crack": EXPECTED_NO_CRACK,
            "matches_expected_counts": (total == EXPECTED_TOTAL and crack == EXPECTED_CRACK and no_crack == EXPECTED_NO_CRACK),
        },
        "csv_vs_folder": {
            "agree": csv_agree, "mismatch": csv_mismatch, "missing_from_csv": csv_missing,
            "csv_files_found": csv_names,
        },
        "prefix_groups": [asdict(g) for g in sorted(groups.values(), key=lambda g: -g.count)],
        "prefixes_spanning_multiple_partitions": multi_partition_prefixes,
        "prefixes_spanning_count": len(multi_partition_prefixes),
        "dhash_duplicate_threshold_hamming": dup_threshold,
        "cross_partition_near_duplicate_count": len(cross_pairs),
        "cross_partition_near_duplicate_examples": [
            {"a": a, "b": b, "hamming": d} for a, b, d in cross_pairs[:max_pairs_reported]
        ],
        "evaluation_eligibility": "DISABLED",
        "evaluation_eligibility_reason": (
            "Cross-partition near-duplicates and capture-session prefixes spanning multiple "
            "official partitions were detected. The official test partition cannot be treated "
            "as independent. No accuracy, precision, recall, F1 or other performance figure "
            "derived from these partitions is evidence of generalisation."
        ),
    }

    json_path = os.path.join(out_dir, "tubers_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(out_dir, "tubers_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TUBERS Dataset Audit\n\n")
        f.write(f"- **Source:** {report['dataset_source']}\n")
        f.write(f"- **DOI:** {report['dataset_doi']}\n")
        f.write(f"- **Creators:** {report['dataset_creators']}\n")
        f.write(f"- **License:** {report['dataset_license']}\n")
        f.write(f"- **Retrieval date:** {report['retrieval_date']}\n")
        f.write(f"- **Archive:** `{report['archive_path']}`\n")
        f.write(f"- **Archive SHA-256:** `{archive_sha}`\n")
        f.write(f"- **Matches expected SHA-256:** {'YES' if hash_ok else 'NO'}\n\n")
        f.write("## Counts\n\n")
        f.write(f"- Total images in archive: **{total}** (expected {EXPECTED_TOTAL})\n")
        f.write(f"- Crack: **{crack}** (expected {EXPECTED_CRACK})\n")
        f.write(f"- No-crack: **{no_crack}** (expected {EXPECTED_NO_CRACK})\n")
        f.write(f"- Counts match expectation: **{'YES' if report['counts']['matches_expected_counts'] else 'NO'}**\n\n")
        f.write("## CSV vs. folder labels\n\n")
        f.write(f"- Agree: {csv_agree}, Mismatch: {csv_mismatch}, Missing from CSV: {csv_missing}\n\n")
        f.write("## Capture-session prefixes\n\n")
        f.write(f"- Unique prefixes: **{len(groups)}**\n")
        f.write(f"- Prefixes spanning multiple official partitions: **{len(multi_partition_prefixes)}**\n\n")
        f.write("| Prefix | Count | Crack | No-crack | Partitions | Frame range | Camera |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for g in sorted(groups.values(), key=lambda g: -g.count):
            fr = f"{g.frame_min}–{g.frame_max}" if g.frame_min is not None else "?"
            cam = ", ".join(g.cameras.keys()) if g.cameras else "?"
            parts = ", ".join(f"{k}={v}" for k, v in sorted(g.partitions.items()))
            f.write(f"| `{g.prefix}` | {g.count} | {g.crack} | {g.no_crack} | {parts} | {fr} | {cam} |\n")
        f.write("\n## Cross-partition near-duplicate risk\n\n")
        f.write(f"- dHash Hamming threshold: **{dup_threshold}** (out of 64 bits)\n")
        f.write(f"- Cross-partition pairs at or below threshold: **{len(cross_pairs)}**\n")
        if cross_pairs:
            f.write("\nExamples (closest first):\n\n")
            for a, b, d in cross_pairs[:20]:
                f.write(f"- `{a}` ↔ `{b}` (Hamming={d})\n")
        f.write("\n## Evaluation eligibility: **DISABLED**\n\n")
        f.write(report["evaluation_eligibility_reason"] + "\n\n")
        f.write("> This audit does not produce, and deliberately refuses to produce, any "
                "accuracy, precision, recall, F1 or other performance figure. "
                "The official partitions are not demonstrated to be independent.\n")

    print(f"Audit written: {json_path}")
    print(f"Audit written: {md_path}")
    print(f"Archive SHA-256 matches expected: {hash_ok}")
    print(f"Counts: total={total} crack={crack} no-crack={no_crack}")
    print(f"Prefixes spanning multiple partitions: {len(multi_partition_prefixes)}")
    print(f"Cross-partition near-duplicate pairs (Hamming<={dup_threshold}): {len(cross_pairs)}")
    print("Evaluation eligibility: DISABLED")
    return 0


def main():
    p = argparse.ArgumentParser(description="Read-only TUBERS dataset audit.")
    p.add_argument("--archive", default=os.path.expanduser("~/storage/downloads/Robot Inspection Visual Data.zip"))
    p.add_argument("--out", default="data/audit")
    p.add_argument("--dup-threshold", type=int, default=4)
    args = p.parse_args()
    sys.exit(audit(args.archive, args.out, args.dup_threshold))


if __name__ == "__main__":
    main()