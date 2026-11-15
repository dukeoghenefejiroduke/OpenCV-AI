# Changelog

All notable changes to CementSight AI will be documented in this file.

## [v1.3.0] - 2026-11-15
### Added
- Phase A: Asset tracking gaps closed (comparable observations >=3, range with "assumes similar framing"), explainability panel derived from real stage counts, and multilingual string tables (Pidgin/French, disabled, "needs native review").
- Phase B: Challenge pack evaluation executed on 24 test images with per-category Wilson 95% Confidence Intervals; `docs/known_failure_modes.md` published.
- Phase C: Evidence-gated numpy logistic regression classifier with group-aware cross-validation, probability calibration, sealed lockbox, dataset manifest export, and labeled photo contribution queue.
- Phase D: Product hardening including security regression test suite, EXIF GPS stripping, delete-my-data action with confirmation audit log, data retention settings, consent text, and offline service worker.
- Phase E: Workflow polish including CSV export of inspection history, per-asset history views, print-ready supervisor reports with blank signature and date lines, and rule-based plain-language summaries.
- Phase F: Lab anomaly detectors, Protocol v3 evaluation with sealed seeds 3001-3040 against pre-declared bounds.
- Phase G: Hosted mode enterprise design and configuration documentation (token auth, HTTPS, rate limits, upload caps, per-user storage, Docker/Lambda notes, marked "not deployed, not verified").
- Phase H: System Card, validation status, known limitations, dynamic metrics summary, attribution (TUBERS CC BY 4.0), and 3-minute pitch outline.

### Security & Integrity
- Frozen OpenCV vision pipeline (`vision/inspection.py`) hash-pinned and untouched.
- Stdlib + numpy + OpenCV only; strict data separation preventing challenge-pack leakage into tuning.
