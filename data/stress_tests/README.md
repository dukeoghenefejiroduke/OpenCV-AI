# CementSight AI — Stress Tests & Known Boundary Conditions

This directory houses dedicated stress tests used to probe the hard physical boundaries and limitations of CementSight AI's classical vision detectors. Stress tests are strictly segregated from real-world field validation datasets.

---

## ST-01: Severe Dark-Brown / Black Pitted Corrosion (Stress Test)

- **Image Path:** `data/stress_tests/ST-01_severe_dark_rust.jpg`
- **Provenance:** Stock imagery reference (Vecteezy watermark present; ineligible for field validation).
- **Asset Material:** Heavily corroded steel column / industrial tank.
- **Visual Condition:** Severe corrosion featuring dark-brown / black pitted rust at the top (30%), a heavy rust band at the bottom (15%), a large center-left rust patch, and numerous orange rust streaks.

### Quantitative Audit & Findings
- **Box Count Performance:** 7 True Positives, 0 False Positives (Precision: 100% on detected spots).
- **Area Recall:** ~20% (Missed top 30% black pitted rust, bottom rust band, and center-left large rust patch; successfully boxed only early-stage orange surface streaks).
- **Bounding Box Over-Merging:** COR-001 over-merged multiple discrete streaks and clean metal regions.

### Failure Mechanism & Technical Rationale
- **HSV Color-Space Limitation:** The production corrosion pipeline (`vision/inspection.py`) relies on classical HSV color segmentation (`[5, 50, 50]` to `[25, 255, 255]`) tuned specifically for early-stage, light orange-brown surface rust on steel.
- **Dark Rust Blindspot:** Severe dark-brown and black pitted corrosion exhibits low luminance ($V$) and high saturation ($S$) that overlap with shadows, dark soil, and wet wood in HSV color space. Widening HSV to capture black rust would immediately cause catastrophic false positives on shadows and non-metallic backgrounds.

### Architectural Decision & Future Work
- **Protected Pipeline Integrity:** To preserve precision on early-stage screening, production HSV thresholds **must not** be hacked or widened.
- **Future Work:** Severe dark-rust detection requires advanced texture, roughness, and pitting analysis (e.g., local contrast or gradient texture descriptors) as a separate experimental detector rather than a modification to the protected production pipeline.
