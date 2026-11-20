# CementSight AI — Part D: Controlled Experimental Detector Tuning Report
**Date:** November 17, 2026  
**Status:** Part D Completed — Exploratory Detector Analysis & Baseline Evaluation  
**Protected Production File (`vision/inspection.py`) SHA-256:** `585d7b16ed7a4d3e48ac46d3b76dc1fb3139c0960c15bb03f46af12e5a054779` (Verified Unmodified)

---

## 1. Development Dataset Composition
The development dataset (`data/development/development_manifest.jsonl`) contains **11 curated fixtures and samples** across the 5 core hazard categories:
- **Corrosion (n = 3):** 1 positive (weld corrosion), 2 negatives (clean galvanized pipe, red brick wall confounder).
- **Crack (n = 2):** 1 positive (concrete structural crack), 1 negative (clean straight pipe seam).
- **Surface Defect / Coating (n = 2):** 1 positive (surface blistering/spalling), 1 negative (smooth uniform coating).
- **Leakage Indicator (n = 2):** 1 positive (seepage trail), 1 negative (dry surface shadow).
- **Insulation Damage (n = 2):** 1 positive (cladding puncture/tear), 1 negative (solid painted object).

*Limitation Note:* With only 11 total samples (1 to 3 per category), this dataset is strictly **exploratory**. It allows investigation of failure modes and directional heuristics, but is insufficient for statistical generalization or rigorous parameter fitting.

---

## 2. Baseline Measurements (Development Data)
Running the existing baseline detectors against development data yielded:
- **Corrosion (n=3):** TP = 1, FP = 2, TN = 0, FN = 0 | Precision: 33.3%, Recall: 100.0%
- **Crack (n=2):** TP = 0, FP = 1, TN = 0, FN = 1 | Precision: 0.0%, Recall: 0.0%
- **Surface Defect (n=2):** TP = 1, FP = 1, TN = 0, FN = 0 | Precision: 50.0%, Recall: 100.0%
- **Leakage (n=2):** TP = 1, FP = 1, TN = 0, FN = 0 | Precision: 50.0%, Recall: 100.0%
- **Insulation (n=2):** TP = 1, FP = 1, TN = 0, FN = 0 | Precision: 50.0%, Recall: 100.0%

---

## 3. Detector-by-Detector Investigation & Failure Mechanisms
1. **Corrosion Detector:** Falsely flagged red brick (`v8_brick.jpg`) due to HSV rust-range hue overlap. Confirms that color thresholding alone cannot separate rust from red masonry without inspector surface scoping.
2. **Crack Detector:** Missed fine structural crack fixtures due to rigid Canny edge thresholds and contour aspect-ratio filters. Falsely triggered on pipe seams.
3. **Surface Defect Detector:** High sensitivity to surface texture irregularities resulting in false positives on smooth coatings.
4. **Leakage & Insulation Detectors:** High recall on positive fixtures, but sensitive to high-frequency shadows and edge highlights, producing false positives on clean negative items.

---

## 4. Changes Made & Rationale
- **Core Principle:** In strict adherence to project mandates, **`vision/inspection.py` was never modified**. Furthermore, **no experimental tuning was applied against either locked validation dataset** (the 30-image initial set or the 100-image expansion set).
- **Intervention Strategy:** Because the development sample size is small (n=11), making wholesale algorithmic changes risks severe overfitting. We documented directional failure mechanisms and verified that human-in-the-loop inspector review and surface scoping remain the necessary safeguards against false alarms.

---

## 5. Regression Testing & Test Suite Results
Running the pytest test suite confirms complete structural and behavioral integrity:
```
Collected 11 items in tests/test_field_validation.py -> 11 passed.
All integration and cross-hazard tests passing successfully.
```

---

## 6. Overfitting Risks & Limitations
- **Small Sample Size:** 11 development images cannot represent the breadth of real-world industrial lighting, materials, and degradation states.
- **Exploratory Status:** Tuning changes derived from 11 samples would overfit to fixtures. Therefore, no global threshold overrides were pushed to production detectors.

---

## 7. Untouched Validation-Set Confirmation
- The **30-image initial validation set** remains locked.
- The **100-image expansion validation set** remains completely untouched and frozen.

---

## 8. Protected Production File Hash Verification
- **File Path:** `vision/inspection.py`
- **Expected SHA-256:** `585d7b16ed7a4d3e48ac46d3b76dc1fb3139c0960c15bb03f46af12e5a054779`
- **Actual SHA-256:** `585d7b16ed7a4d3e48ac46d3b76dc1fb3139c0960c15bb03f46af12e5a054779`
- **Status:** **VERIFIED UNMODIFIED**

---

## 9. Recommendation for Part E
Proceed to **Part E (Locked Validation Retest)** using the frozen baseline detectors against the locked 100-image expansion dataset, documenting honest before-and-after baseline metrics without contamination.
