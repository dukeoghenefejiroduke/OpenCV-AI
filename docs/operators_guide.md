# CementSight AI — Multi-Hazard Operator's Guide

This guide details the five modular vision detectors implemented in CementSight AI v1.2.0, outlining their inputs, screening methodology, common false positives, operational limitations, and mandatory human-review protocols.

---

## Safety & Governance Notice

All experimental screening detectors in CementSight AI operate under strict safety governance:
1. **Isolation from Authoritative Decisions:** Experimental detectors produce provisional findings marked as `PENDING_REVIEW` and do not alter authoritative corrosion coverage percentages, thresholds (`PASS`, `WARNING`, `CRITICAL`), or automated safety decisions.
2. **Mandatory Human Verification:** No experimental screening candidate can trigger automated work-order dispatch or asset safety-status changes. All findings require human inspection by certified plant engineers.
3. **Screening Estimates Only:** Detectors provide classical computer-vision screening estimates based on pixel heuristics and gradients; they do not provide certified physical measurements, thermal efficiency metrics, or definitive structural failure confirmation.

---

## 1. Classical Corrosion & Rust Inspector (`corrosion_detector`)
- **Defect Category:** `corrosion`
- **Supported Inputs:** BGR numpy array or valid image path (minimum 128x128 resolution).
- **Screening Method:** HSV color space thresholding for rust/orange-brown tones combined with classical Canny edge foreground ROI segmentation and morphological noise filtering.
- **Example Detections:** Rust patches on industrial pipes, structural steel corrosion, oxide staining on storage tanks.
- **Common False Positives:** Non-rust orange coatings, surface grime, reflected lighting glare, and discolored paint.
- **Operational Limitations:** Sensitive to lighting variations and hue saturation; screen estimate rather than certified physical material loss.
- **Human Review:** Warning and Critical levels trigger mandatory supervisor review before maintenance scheduling.

---

## 2. Classical Structural Crack Screening Detector (`crack_screening_detector`)
- **Defect Category:** `crack`
- **Supported Inputs:** BGR numpy array or valid image path (minimum 128x128 resolution).
- **Screening Method:** Gaussian smoothing, Canny edge detection, morphological closing, and contour/perimeter geometric filtering for elongated linear features.
- **Example Detections:** Linear surface cracks, structural concrete fractures, linear joint gaps.
- **Common False Positives:** Scratch marks, painted seams, weld lines, background textures, and high-contrast shadows.
- **Operational Limitations:** Experimental screening; high false-positive rate on industrial background texture and weld seams. Uncalibrated pixel measurements.
- **Human Review:** All crack candidates require manual inspection to distinguish surface scratches from structural cracking.

---

## 3. Classical Surface Defect Screening Detector (`surface_defect_detector`)
- **Defect Categories:** `coating_damage` (blistering) & `spalling` (concrete spalling)
- **Supported Inputs:** BGR numpy array or valid image path (minimum 128x128 resolution).
- **Screening Method:** Adaptive thresholding, local intensity variation, Gaussian blur, morphological filtering, circularity filtering ($4\pi \text{area}/\text{perimeter}^2$), and aspect ratio analysis.
- **Example Detections:** Rounded paint blisters (coating damage) and irregular chipped concrete patches (spalling).
- **Common False Positives:** Surface dirt, reflections, localized stains, and background texture variations.
- **Operational Limitations:** Blob geometry (circularity/irregularity) does not confirm chemical blisters or structural spalling. Experimental screening only.
- **Human Review:** Mandatory physical inspection required to confirm coating integrity or concrete spalling.

---

## 4. Classical Leakage & Seepage Indicator Screening Detector (`leakage_detector`)
- **Defect Category:** `leakage`
- **Supported Inputs:** BGR numpy array or valid image path (minimum 128x128 resolution).
- **Screening Method:** Grayscale conversion, Gaussian blurring, adaptive thresholding, vertical morphological streak structuring (`(3, 11)` kernel), and contour aspect-ratio screening.
- **Example Detections:** Vertical/diagonal staining streaks, fluid run marks, down-drainage discoloration patches.
- **Common False Positives:** Structural shadows, pipe support brackets, grime streaks, and background paint gradients.
- **Operational Limitations:** Dark vertical streaks may be shadows or oil marks rather than active fluid leakage. RGB photographs cannot establish active leak rate or fluid composition.
- **Human Review:** Marked as `potential_leakage_indicator` and `possible_seepage_staining`; requires manual inspection and physical verification.

---

## 5. Classical Insulation Damage & Cladding Screening Detector (`insulation_detector`)
- **Defect Category:** `insulation_damage`
- **Supported Inputs:** BGR numpy array or valid image path (minimum 128x128 resolution).
- **Screening Method:** Canny edge detection, adaptive thresholding, morphological closing, contour area and solidity analysis ($\text{area}/\text{hull\_area} < 0.85$).
- **Example Detections:** Missing or displaced cladding panels, open seams in insulation jackets, and exposed underlying material.
- **Common False Positives:** Normal construction joints, overlapping cladding seams, mechanical support straps, and brackets.
- **Operational Limitations:** Ordinary RGB photographs cannot establish thermal conductivity, insulation efficiency, operating temperature, or internal insulation degradation.
- **Human Review:** Requires manual inspection and thermal/contact verification; provisional screening status only.
