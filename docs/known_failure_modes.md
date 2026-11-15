# Known Failure Modes & Challenge Pack Evaluation Report

This document records the empirical results and known failure modes of CementSight AI based on evaluation against the challenge pack (`data/challenge_pack`) evaluated on November 15, 2026 (`data/outputs/challenge_report_20261115_171520.json`).

## Executive Summary
The challenge pack comprises 24 test images across 12 distinct categories (2 items per category). **Note:** 22 of the 24 photos were previously used for development or earlier testing; therefore, these results serve as a regression check for known failure modes rather than an independent accuracy or generalization benchmark.

Each item was evaluated using the default OpenCV corrosion inspection pipeline and the experimental adaptive rust detector (`RustDetector`).

## Overall Sensitivity and Specificity Summary

| Detector | Sensitivity (Rust Photos Flagged, n=12) | Specificity (Clean Photos Passed, n=12) | Overall Correct (n=24) |
| :--- | :---: | :---: | :---: |
| **Default Detector** | 0.42 (5/12) | 0.58 (7/12) | 12 / 24 (50%) |
| **Rust Detector (`RustDetector`)** | 0.75 (9/12) | 0.33 (4/12) | 13 / 24 (54%) |

*Trade-off analysis:* While the experimental rust detector achieves higher sensitivity on corroded samples (9/12), it suffers from low specificity (falsely flagging 8 out of 12 clean photos, including silver, orange, and red brick categories), demonstrating that higher sensitivity comes at the cost of excessive false alarms.

## Per-Category Evaluation Results

| Category | Total (n) | Default Detector Correct | Rust Detector Correct |
| :--- | :---: | :---: | :---: |
| **blur** | 2 | 2 | 1 |
| **thin_corrosion** | 2 | 1 | 2 |
| **orange** | 2 | 1 | 0 |
| **wood** | 2 | 0 | 1 |
| **dark** | 2 | 1 | 1 |
| **cluster** | 2 | 0 | 2 |
| **rust_shadow** | 2 | 1 | 2 |
| **joint** | 2 | 1 | 1 |
| **silver** | 2 | 2 | 0 |
| **red_brick** | 2 | 0 | 0 |
| **clean** | 2 | 2 | 2 |
| **rust** | 2 | 1 | 1 |

## Known Failure Modes Identified
1. **False Positives on Non-Metallic Textured & Colored Surfaces (Wood, Red Brick, Orange, Silver):**
   - Warm color distributions and reflective finishes overlap with chromatic thresholds, leading to high false-positive rates when sensitivity is increased.
2. **Low Illumination and Shadows (Rust in Shadow):**
   - High-contrast shadows suppress color saturation, affecting default detector sensitivity (1/2 correct) while the rust detector flagged both, albeit at the expense of specificity elsewhere.
3. **Thin/Distributed Corrosion:**
   - Discontinuous or micro-pitting corrosion challenges default thresholding (1/2 correct).
4. **Clustered Defect Regions:**
   - Dense defect clusters require careful morphological tuning to balance detection and false alarms.
