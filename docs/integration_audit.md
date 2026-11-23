# DeltaTwin Integration Audit & Architecture Plan

## 1. Executive Summary & Audit Answers
This document outlines the architecture, data schemas, phased roadmap, and regression test plan for extending CementSight AI into DeltaTwin.

### Audit Answers:
1. **`/api/inspect` Image Path Security & Test Status:**
   - *Previous state:* In `tools/server.py`, absolute or `..` paths were checked with `if ".." in normalized_img_path or os.path.isabs(normalized_img_path): if not os.path.exists(image_path):`. This meant that if an absolute or `..` path *existed* on disk, it bypassed the check and allowed arbitrary file inspection/read.
   - *Fix / Verification:* Replaced with strict absolute/`..` rejection and directory allow-listing (`data/` and `app/` only), accompanied by a dedicated test in `tests/test_dashboard.py` (`test_7_api_inspect_absolute_path_rejected`) verifying that existing absolute paths return 400.
2. **Orchestrator Contract-Rejected Findings:**
   - *Previous state:* In `vision/orchestrator.py`, when a finding failed `validate_finding()`, the block executed a bare `pass`, silently dropping invalid findings.
   - *Fix / Verification:* Updated to record `contract_rejection_count` and `rejected_findings_summary` in detector results and summaries.

---

## 2. Module Inventory & Reuse Notes
- **`workspace/` (`store.py`, `models.py`, `prioritization.py`, `report.py`, `backup.py`):** Reused entirely as the immutable SQLite storage, review state machine, and prioritization engine.
- **`app/digital_twin.py`:** Reused as the asset hierarchy and relational node mapping foundation.
- **`app/forecasting.py`:** Reused for degradation velocity and time-to-critical projections (requiring $\ge 3$ dated observations).
- **`workspace/prioritization.py`:** Reused for human-in-the-loop transparent risk scoring (gated by `crack_enabled_for_priority=False` and requiring `confirmed` status).

---

## 3. Overlaps & Proposed Packages & Simulation Isolation
Rather than building a parallel application, existing modules will be extended into dedicated backend packages:
- **`telemetry/`:** Time-series telemetry schema with a mandatory `provenance` field (`simulated` | `imported_real`), plus a seeded telemetry simulator injecting physical faults.
- **`anomaly/`:** Transparent statistical anomaly detectors (rolling robust median/MAD Z-score, EWMA, CUSUM). No black-box machine learning models.
- **`twin/`:** Network graph model (nodes, segments, assets) stored as JSON, aggregating health scores.
- **`risk/`:** Rule-based fusion engine combining *confirmed* visual findings with telemetry anomalies into a transparent risk band with explicit rationale.

### Simulation Namespace & Isolation
- **Strict Isolation:** Simulated telemetry data lives in a dedicated simulation namespace/database table and is tagged with `provenance: "simulated"`.
- **Priority Firewall:** Simulated telemetry and simulated anomalies are strictly prohibited from feeding real asset history, priority scoring, or generating automated work orders. They are restricted solely to the **Condition Alerting Demo** views.

---

## 4. Extended Telemetry Data Schema (`telemetry/schema.py`)
```json
{
  "sensor_id": "PT-102",
  "asset_id": "PIPE-01",
  "timestamp": "2026-11-13T12:00:00Z",
  "metric": "pressure_drop_kpa",
  "value": 14.2,
  "provenance": "simulated",
  "scenario_id": "scen_leak_01",
  "ground_truth_fault": true,
  "simulator_version": "v1.0",
  "seed": 42,
  "quality_flag": "ok",
  "units": "kPa"
}
```
*(Note: `provenance`, `timestamp`, `scenario_id`, `simulator_version`, and `seed` are mandatory fields).*

---

## 5. Implemented Unified Architecture & `/api/v2` Routes
Both runtimes (`server.ts` + `deltaTwinEngine.ts` on Node/Express port `3000`, and `tools/server.py` + `telemetry/`, `anomaly/`, `twin/`, `risk/` in Python) expose matching contracts and strict simulation isolation:
- `GET /api/v2/telemetry/:assetId`: Retrieves `[SIMULATED]` (`SIM-*`) or `[IMPORTED_REAL]` telemetry history and anomaly alerts.
- `POST /api/v2/telemetry/simulate`: Seeded Mulberry32 + Box-Muller AR(1) 4-metric simulator (`pressure_kpa`, `flow_m3h`, `temperature_c`, `vibration_mm_s`) with `linear`, `exponential`, and `weibull` fault ramps (`leak`, `bearing_wear`, `blockage`, `overheating`). **Limits:** 10KB payload cap, 10 req/min rate limit, 5,000-record retention cap in `data/simulation/node_telemetry_sim.jsonl`, mandatory `[SIMULATED]` badge & `SIM-` namespace.
- `POST /api/v2/telemetry/import` & `POST /api/v2/telemetry/upload`: Imports real CSV telemetry tagged `provenance: "imported_real"` (`[IMPORTED_REAL]`) with mandatory `source_url`, `license`, and `retrieved_date` verification; blocks `SIM-` asset IDs.
- `GET /api/v2/twin/network` (`?mode=real|sim|all`) & `POST /api/v2/twin/network`: DAG digital twin validation, DFS cycle rejection, `CRITICAL > WARNING > HEALTHY` propagation, and `SIM` vs real asset namespace isolation.
- `POST /api/v2/risk/fuse` & `GET /api/v2/intelligence/overview`: Unified predictive risk fusion combining confirmed visual inspection findings, corrosion degradation velocity (`>= 3` dated observations), and non-simulated telemetry anomalies (`dispatched: false` strictly enforced; simulation firewall strips simulated alerts from real assets).
- **Atomic Workspace Persistence:** `data/workspace/node_state.json` (plus `POST /api/workspace/backup` and path-confined `POST /api/workspace/restore` under `data/backups/`).

---

## 6. Measurable Acceptance Criteria & Held-Out Tuning
- **Detection Benchmark:** Every injected fault type must be detected within $N \le 5$ samples across fixed evaluation seeds.
- **False Alarm Bound:** False-alarm rate must not exceed $\le 2$ false alarms per 1,000 steady-state samples.
- **Detection Delay:** Measured in samples from fault inception to alert trigger.
- **Train/Test Separation (Held-Out Scenarios):** Anomaly detectors and thresholds are tuned on training seeds/scenarios and evaluated exclusively on held-out test seeds/scenarios to prevent data leakage and overfitting.
- **Terminology:** All synthetic telemetry alerting views carry explicit `[SIMULATED]` badges; predictive estimates include explicit screening disclaimers (`dispatched: false`).

---

## 7. Implementation Status (Phases 0–8 Complete)
- **Phase 0:** Baseline regression guard & test suite validation (`pytest -v` + `tsc --noEmit`).
- **Phase 1:** Telemetry schema with mandatory provenance (`telemetry/schema.py`, `deltaTwinEngine.ts`).
- **Phase 2:** Seeded 4-metric AR(1) simulator with held-out scenario splitting (`telemetry/simulator.py`, `deltaTwinEngine.ts`).
- **Phase 3:** Transparent anomaly detectors (Median/MAD Robust Z-score, EWMA, CUSUM) with false-alarm and delay benchmarks (`anomaly/detectors.py`, `deltaTwinEngine.ts`).
- **Phase 4:** DAG Digital Twin network graph and upstream-to-downstream health propagation (`twin/graph.py`, `deltaTwinEngine.ts`).
- **Phase 5:** Unified predictive maintenance risk fusion with simulation firewall (`risk/engine.py`, `deltaTwinEngine.ts`, `tests/test_deltatwin_unified.py`).
- **Phase 6:** Offline SVG Engineering Intelligence dashboard (`8. Engineering Intelligence` in `app/dashboard.html`) & `/api/v2` endpoints with `[SIMULATED]` / `[IMPORTED_REAL]` badges and storage retention caps.
- **Phase 7:** Import of real CSV telemetry with mandatory source URL, license, and retrieval date verification.
- **Phase 8:** Atomic `node_state.json` persistence, backup/restore, and unified documentation.

---

## 8. Regression Test Plan
- **Inspection Hash Pin:** Pin SHA-256 hash of `vision/inspection.py` (`1c87c5f05d28720dbbc42d10d49e66ec6b3b27384825297b03685af01cf00d43`).
- **Corrosion Golden Values:** Assert corrosion detector outputs `9.3%` / `4` regions / `WARNING` on reference image `data/pipe_corrosion.jpg`.
- **DeltaTwin Predictive Fusion & Firewall:** Verify simulation firewall, `<3` observation velocity gating, multimodal escalation, and simulator->detector->twin->fusion flow (`tests/test_deltatwin_unified.py`).
- **API & Security Tests:** Validate response structures, path traversal rejection, zero external CDN dependencies, and HTML escaping.
- **Test Execution:** Run `pytest -v` and `npm run lint` (`tsc --noEmit`).

