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

## 5. API Additions (`/api/v2`) & Safety Limits
- `GET /api/v2/telemetry/<asset_id>`: Retrieves telemetry history with provenance tags.
- `POST /api/v2/telemetry/simulate`: Triggers simulation of fault scenarios. **Limits:** Enforces payload size limit (max 10KB), rate-limiting (max 10 requests/minute), and local storage retention cap (max 5,000 telemetry records per device to prevent mobile storage exhaustion).
- `GET /api/v2/twin/network`: Returns topological network graph and asset health summary.
- `POST /api/v2/risk/fuse`: Fuses visual findings and telemetry anomalies into a transparent risk assessment.
- **UI Requirement:** Every `/api/v2` response, chart, report, and export derived from simulated telemetry must prominently display a **SIMULATED** badge.

---

## 6. Measurable Acceptance Criteria & Held-Out Tuning
- **Detection Benchmark:** Every injected fault type must be detected within $N \le 5$ samples across fixed evaluation seeds.
- **False Alarm Bound:** False-alarm rate must not exceed $\le 2$ false alarms per 1,000 steady-state samples.
- **Detection Delay:** Measured in samples from fault inception to alert trigger.
- **Train/Test Separation (Held-Out Scenarios):** Anomaly detectors and thresholds are tuned on training seeds/scenarios and evaluated exclusively on held-out test seeds/scenarios to prevent data leakage and overfitting.
- **Terminology:** All telemetry alerting features are explicitly labeled **"Condition Alerting Demo"** (no predictive maintenance claims).

---

## 7. Phased Roadmap
- **Phase 0:** Baseline regression guard & test suite validation (`131 tests` passing).
- **Phase 1:** Telemetry schema with mandatory provenance (`telemetry/`).
- **Phase 2:** Seeded simulator with held-out scenario splitting.
- **Phase 3:** Transparent anomaly detectors (Z-score, EWMA, CUSUM) with false-alarm and delay benchmarks.
- **Phase 4:** Twin network graph and health aggregation (`twin/`).
- **Phase 5:** Rule-based risk fusion with simulation isolation (`risk/`).
- **Phase 6:** Dashboard engineering tab & `/api/v2` endpoints with mandatory **SIMULATED** badges and storage retention caps.
- **Phase 7:** Import of real CSV telemetry (BattLeDIM / UCI hydraulic sets, with license verification and `provenance: "imported_real"`).
- **Phase 8:** Demo script, metrics page, and release documentation.

---

## 8. Regression Test Plan
- **Inspection Hash Pin:** Pin SHA-256 hash of `vision/inspection.py`.
- **Corrosion Golden Values:** Assert corrosion detector outputs exactly 6.56% / 11 regions / WARNING on reference image `data/pipe_corrosion.jpg`.
- **API Snapshot Tests:** Validate response structures of all endpoints.
- **Offline & XSS Tests:** Verify zero CDN dependencies and XSS sanitization (`html.escape`).
- **Test Execution Count:** Full test suite run (`python -m pytest`) collects and passes **131 tests** successfully.
