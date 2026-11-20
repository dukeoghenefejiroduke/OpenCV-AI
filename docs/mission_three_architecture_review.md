# CementSight AI — Mission Three Architecture Review

**Date:** November 18, 2026  
**Scope:** Architectural review of inspection pipelines, material scoping integration, workspace storage, and review state machine.

## 1. Executive Summary
Mission Three establishes a rigorous material-aware inspection workflow building upon Mission Two's experimental findings. While classical HSV corrosion detection achieves 100% recall on steel assets, it suffers from false positives on non-steel reddish backgrounds (brick, wood, soil). Mission Three formalizes how material declarations and provenance are captured, validated, and stored without compromising safety or masking independent hazards.

## 2. Core Architecture Modules Inspected
- **`vision/inspection.py`:** Core OpenCV inspection pipeline (retained in its protected v1.3 state with SHA-256 integrity verification).
- **`workspace/store.py` & `workspace/models.py`:** SQLite-backed persistence layer supporting asset CRUD, inspection records, finding logs, and material/surface attributes (`pipe_material`, `surface`).
- **`tools/server.py` & `app/dashboard.html`:** API endpoints (`/api/workspace/inspection`) and single-file offline UI handling asset registration, material dropdowns, and inspection submission.
- **`workspace/report.py`:** HTML supervisor report generation incorporating scoping notes and limitations banners.

## 3. Architectural Safety Constraints
1. **No Silent Suppression:** Designating a surface as non-steel (e.g., concrete, brick, wood, insulation) restricts steel corrosion screening but preserves multi-hazard secondary detectors (cracks, leakage, insulation damage, surface spalling).
2. **Provenance Separation:** Inspector-declared material metadata is strictly distinguished from independently verified material records.
3. **Immutable History:** Material attribute updates or subsequent inspections never overwrite past inspection audit records.
