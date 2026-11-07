# TUBERS Dataset Provenance & Audit Report

## Dataset Overview
- **Name:** TUBERS: Visual Data from Water Pipe Inspection
- **Source:** [Zenodo Record 20136698](https://zenodo.org/records/20136698)
- **DOI:** [10.5281/zenodo.20136698](https://doi.org/10.5281/zenodo.20136698)
- **License:** CC BY 4.0
- **Creators:** Bendabl; University of Essex
- **Retrieval Date:** 2026-11-05
- **Archive SHA-256:** `e1d7d78a903b2ef9454726d0938d0ff9a96348e9dcd73e8185bd0a0d7ad774b9`

## Internal Structure & Audit Findings
- **Total Images:** 930 (418 crack, 512 no-crack) — matches expected counts exactly.
- **Internal Archive Layout:** Nested under `Robot Inspection Visual Data/` root directory containing `train/`, `val/`, and `test/` partitions split into `crack/` and `no-crack/` subdirectories.
- **Prefixes Spanning Multiple Partitions:** 16 capture-session prefixes appear across multiple dataset partitions (train/val/test).
- **Cross-Partition Near-Duplicates:** 23,981 cross-partition pairs detected at Hamming distance $\le 4$ using 64-bit dHash.
- **Evaluation Eligibility:** **DISABLED**. Because sequential video frames and capture-session near-duplicates leak across train, validation, and test partitions, the dataset is not demonstrably independent. Consequently, benchmark accuracy, precision, recall, and F1 metrics derived from this split are structurally invalidated for real-world generalization claims.
