# CementSight AI — 3-Minute Pitch Outline

## 1. The Challenge (0:00 - 0:45)
- Industrial cement plants and pipeline networks require frequent, reliable surface corrosion screening.
- Field inspectors need offline-first, edge-capable tools that work without cloud connectivity, protect data privacy, and enforce strict human-in-the-loop accountability.
- Existing tools lack verifiable security boundaries, transparent explainability, and rigorous empirical validation.

## 2. The Solution: CementSight AI v1.3.0 (0:45 - 2:00)
- **Edge-First Architecture:** Operates entirely locally using Python stdlib, NumPy, and OpenCV, requiring zero external cloud APIs or proprietary ML frameworks.
- **Frozen OpenCV Inspection Pipeline:** Proven 11-stage computer vision pipeline (`vision/inspection.py`) achieving deterministic corrosion estimation (e.g., golden corrosion baseline: 6.56% / 11 regions / WARNING).
- **Robust Asset Tracking & Workflow Polish:** Multi-observation comparable tracking (>=3 observations with framing assumptions), CSV inspection history exports, and print-ready supervisor reports with mandatory blank signature and date lines.
- **Enterprise Hardening & Security:** EXIF GPS stripping on stored images, path traversal protection, 25MB payload caps, and an audited "delete my data" action.
- **Evidence-Gated Classifier & Lab Protocols:** Protocol v3 telemetry anomaly simulation across sealed seeds (3001-3040) meeting pre-declared false alarm bounds.

## 3. Validation, Trust & Conclusion (2:00 - 3:00)
- **Rigorously Tested:** Full test suite of 186+ unit tests passing with zero regressions and pinned pipeline hashes.
- **Transparent Limitations:** Fully documented challenge pack evaluation results and known failure modes (e.g., wood and brick false positives).
- **Mandatory Human Sign-Off:** CementSight AI is a screening aid only; all final operational decisions and maintenance approvals remain strictly with qualified human supervisors.
