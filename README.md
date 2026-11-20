> **Research prototype. Screening aid only. Not validated for real-world use.**

# CementSight

CementSight is an industrial visual inspection system for detecting corrosion-like surface conditions on cement/industrial equipment and turning visual findings into actionable maintenance decisions.

---

## Problem

Industrial equipment and cement plants operate in harsh environments where surface degradation and corrosion pose significant structural and operational risks. However, traditional monitoring faces several core challenges:

- **Manual Inspection Dependency:** Visual inspections are often manual, subjective, and difficult to scale.
- **Inconsistent Identification:** Corrosion and degradation can be challenging to identify and quantify consistently across varying lighting and surface conditions.
- **Actionable Traceability:** Inspection findings need to translate reliably into structured maintenance actions rather than isolated image markers.
- **Safety and Oversight:** Automated inspection systems must provide transparent evidence, defensive safety controls, and explicit human oversight before taking critical actions.

*Note: CementSight is a screening and decision-support tool and does not make unsupported claims about financial savings, certified measurement precision, or unverified real-world deployment.*

---

## Solution

CementSight implements a deterministic, multi-stage inspection pipeline that transforms raw equipment imagery into actionable operational intelligence:

```
Image
→ input validation
→ OpenCV inspection
→ structured inspection result
→ agentic decision layer
→ proposed maintenance action
→ human approval when required
```

The system combines classical computer vision (OpenCV) with a deterministic, rule-based agentic decision layer. Visual inspection outputs strictly determine the subsequent operational state without relying on LLMs, randomness, or network calls.

---

## Key Features

- **OpenCV-Based Multi-Hazard Screening:** Robust classical computer vision pipelines for corrosion, structural cracks, surface defects (blistering/spalling), leakage/seepage, and insulation damage.
- **Structural Foreground ROI Detection:** Edge detection and morphological grouping to isolate equipment boundaries.
- **Extent & Noise Filtering:** Rejection of tiny background artifacts and giant frame-spanning background contours.
- **Region Detection & Bounding Boxes:** Accurate bounding box extraction and region counting for identified hazard clusters.
- **Deterministic Agent Decision Layer:** Rule-based classification producing CRITICAL, WARNING, or PASS operational states.
- **Human Approval Enforcement:** Mandatory supervisor review and approval before any CRITICAL maintenance work-order proposal can proceed.
- **Input Validation & Safety Layer:** Dimension sanity checks (minimum 128x128 pixels) to prevent unreliable inspection on undersized or corrupt images.
- **AWS Lambda Adapter:** Ready-to-deploy serverless adapter supporting Function URLs with structured JSON error handling (HTTP 400/200).
- **Reproducible Evaluation Harness:** Automated evaluation script (`tools/evaluate.py`) capturing environment metrics, performance, robustness, and safety evidence.
- **Comprehensive Automated Tests:** Robust unit, integration, multi-hazard, robustness, and validation test cases (run `python -m pytest` to execute).

---

## Agentic Vision Workflow

CementSight maps visual inspection metrics into three distinct operational decision states:

```
CRITICAL
→ proposed immediate emergency maintenance work-order request
→ supervisor approval required

WARNING
→ proposed routine maintenance inspection
→ ongoing monitoring

PASS
→ normal equipment status recorded
→ no maintenance escalation
```

*Important Safety Notice:* The local system proposes actions and requests but **does not actually dispatch** external maintenance work orders or connect to real-world maintenance dispatch systems.

---

## OpenCV 5 Compliance

CementSight adopts a dual-environment architecture to balance local development constraints with advanced cloud capabilities:

- **Local Development (Android / Termux / ARM64):** Runs on compatible local OpenCV 4.x wheels due to native ARM64 compilation constraints.
- **Competition / Cloud Target (GitHub Actions CI / AWS Lambda):** Runs on `opencv-python-headless` 5.0.0.93 with Python 3.11 and NumPy 2.2.3.

The AWS-targeted Linux runtime and OpenCV 5 compliance are configured via GitHub Actions CI workflow and verified locally via the test harness.

---

## AWS Architecture

```
Client / image
|
v
AWS Lambda Function URL
|
v
Lambda Python 3.11
|
+--> OpenCV 5 inspection
|
+--> Agent decision layer
|
v
Structured inspection + proposed action
```

*Status Note:* Live AWS deployment is pending account activation; the Lambda package build and Linux/OpenCV 5 runtime workflow is configured in `.github/workflows/build-lambda.yml`; local test harness passes; live cloud deployment pending account activation.

---

## Current Evaluation

Evaluated against the real industrial test image (`data/pipe_corrosion.jpg`):

| Metric | Result |
| :--- | :--- |
| **Image** | `data/pipe_corrosion.jpg` |
| **Dimensions** | `1920x1080` |
| **Corrosion estimate** | `6.56%` *(computer-vision screening estimate, not certified physical measurement)* |
| **Detected regions** | `11` |
| **Vision recommendation** | `WARNING` |
| **Agent decision** | `WARNING` |
| **Proposed action** | Routine maintenance inspection and ongoing monitoring |
| **Approval required** | `No` |

**Test Suite Summary:** `208/208` automated tests passed successfully across all inspection, agent, multi-hazard, quality gate, robustness, workspace, security, and validation suites.

---

## Vision Pipeline & Quality Gate

CementSight includes an **Image Quality Gate** (`vision/quality_gate.py`) running prior to analysis to assess blur (Laplacian variance), exposure, glare, and contrast. If low quality is detected, detectors still execute but findings are tagged with low confidence and a retake recommendation without silent suppression.
Modular experimental screening detectors (cracks with CLAHE and straight-seam suppression, surface defects / blistering / spalling, leakage, insulation, and optional unvalidated material buildup) feature illumination normalization, rejection auditing, and strict experimental disclaimers.

---

## Robustness & Failure Cases

CementSight has been rigorously tested across 12 robustness and failure-case categories:

- **Clean Synthetic Image:** `PASS` (Corrosion: `0.0%`)
- **Moderate Synthetic Corrosion:** `WARNING` (Corrosion: `7.65%`)
- **High Synthetic Rectangle:** `PASS` (Corrosion: `0.0%`) — *Observed Limitation:* Structural ROI filtering rejects plain artificial rectangular geometry without natural equipment edge gradients.
- **Dark Image:** Handled safely with finite numeric output (`0.0%`).
- **Bright Image:** Handled safely with finite numeric output (`0.0%`).
- **Noisy Image:** Handled deterministically (`0.0%`, `PASS`).
- **Tiny 32x32 Image:** Safely rejected by the input validation safety layer (`INPUT REJECTED / UNSUITABLE`).
- **Invalid / Corrupt Input:** Safely caught and rejected.
- **Missing Input Path:** Safely caught (`FileNotFoundError`).

*Analysis:* The high synthetic rectangle behavior highlights that classical structural ROI segmentation depends on visible scene structure and edge gradients, proving why real industrial imagery (`pipe_corrosion.jpg`) is essential for accurate evaluation.

---

## Safety & Responsible Operation

- **Input Validation:** Enforces minimum resolution thresholds (128x128 pixels) and corrupt-file checks.
- **Defensive Handling:** Robust exception catching and fallback behavior for edge cases.
- **Deterministic Decisions:** Pure rule-based logic ensuring reproducibility without LLM hallucinations or randomness.
- **Human Oversight:** Mandatory approval workflows for CRITICAL findings.
- **No Automatic Dispatch:** Proposes actions without modifying external systems or dispatching real-world work orders.
- **Transparent Limitations:** Surfaces observed limitations rather than hiding edge cases.

---

## Reproduce Locally

Run the following commands from the repository root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run vision tests
PYTHONPATH=. python3 tests/test_inspection.py

# 3. Run agent decision tests
PYTHONPATH=. python3 tests/test_agent_decision.py

# 4. Run robustness tests
PYTHONPATH=. python3 tests/test_robustness.py

# 5. Run input validation tests
PYTHONPATH=. python3 tests/test_validation.py

# 6. Run evaluation harness
PYTHONPATH=. python3 tools/evaluate.py

# 7. Run CLI against real industrial test image
PYTHONPATH=. python3 app/main.py data/pipe_corrosion.jpg
```

---

## Project Structure

```
cementsight-ai/
├── agent/
│   └── decision.py          # Deterministic rule-based agentic decision layer
├── app/
│   ├── cmms.py              # CMMS work-order integration module
│   ├── dashboard.html       # Web dashboard interface
│   ├── digital_twin.py      # Digital twin asset status mapping
│   ├── forecasting.py       # Degradation velocity & time-to-critical projections
│   ├── main.py              # Integrated CLI workflow application
│   └── validation.py        # Input safety validation layer
├── vision/
│   ├── contract.py          # Vision detector interfaces & data contracts
│   ├── corrosion_detector.py # Corrosion inspection pipeline
│   ├── crack_detector.py    # Crack screening detector
│   ├── diagnostics.py       # Vision diagnostic stage recording
│   ├── inspection.py        # Main inspection orchestrator
│   ├── insulation_detector.py # Insulation damage detector
│   ├── leakage_detector.py  # Leakage / seepage indicator detector
│   ├── orchestrator.py      # Multi-hazard orchestrator
│   ├── spalling.py          # Spalling / surface defect detector
│   ├── stream.py            # Streaming inspection pipeline
│   └── surface_defect_detector.py # Surface defect screening detector
├── tests/
│   ├── test_advanced_features.py
│   ├── test_agent_decision.py
│   ├── test_cross_hazard_benchmark.py
│   ├── test_dashboard.py
│   ├── test_diagnostics.py
│   ├── test_field_validation.py
│   ├── test_inspection.py
│   ├── test_insulation_detector.py
│   ├── test_leakage_detector.py
│   ├── test_measurement_reliability.py
│   ├── test_multi_hazard.py
│   ├── test_robustness.py
│   ├── test_safety_isolation.py
│   ├── test_surface_defect_detector.py
│   └── test_validation.py
├── tools/
│   ├── evaluate.py          # Reproducible evaluation harness script
│   ├── generate_diagrams.py # Diagram generator script
│   └── validate_dataset.py  # Dataset validation utility
├── data/
│   ├── pipe_corrosion.jpg   # Real industrial test image (1920x1080)
│   ├── field_validation/    # Field validation dataset & manifest
│   └── outputs/             # Generated evaluation reports and artifacts
├── lambda_handler.py        # AWS Lambda serverless function adapter
├── requirements.txt         # Local development dependencies
├── requirements-lambda.txt  # AWS Lambda deployment dependencies
└── README.md                # Project documentation
```

---

## Validation Status & Known Limitations

- **Crack Detection:** Evaluated on the TUBERS exploratory water-pipe dataset. Results show no demonstrated signal (ROC-AUC 0.3685, 95% CI 0.25–0.53). Crack findings are strictly categorized as "candidate - human review required" and do not contribute to priority or severity scoring by default (`crack_enabled_for_priority=False`).
- **Corrosion Screening:** Primary product focus, evaluated with strict dataset manifests, SHA-256 integrity checks, and sensitivity/specificity metrics with Wilson score confidence intervals.
- **Scene Sensitivity:** Classical computer vision algorithms are sensitive to extreme lighting variations, reflections, and complex background clutter.
- **Screening Estimate:** Corrosion percentages represent computer-vision screening estimates rather than certified physical metallurgical measurements.
- **Human-in-the-Loop:** CRITICAL operational findings strictly require human supervisor review and approval, authenticated with reviewer names and immutable append-only audit trails.

---

## Roadmap

- Live AWS deployment once account activation is completed.
- Public competition demonstration web endpoint.
- Broadened real-world validation dataset across diverse industrial equipment.
- Expansion to additional industrial defect classes (e.g., cracking, spalling).
- Enriched integration connectors for enterprise maintenance management systems with explicit human confirmation.

---

## Competition Notes

CementSight is engineered around:
- OpenCV 5 cloud-target compliance
- AWS-targeted serverless architecture
- Agentic visual decision-making
- Complete reproducibility and automated testing
- Rigorous safety standards and human oversight

---

## Inspection Workspace (Experimental)

CementSight includes an offline, local inspection workspace (`workspace/`) and API routes (`/api/workspace/*`) providing:
- SQLite-backed asset registry and inspection tracking
- Multi-hazard orchestrator execution per inspection
- Finding review workflows (`confirmed`, `rejected`, `needs_followup`)
- Multi-factor maintenance prioritization and risk scoring
- Standalone HTML inspection report generation

---

## Dataset Provenance (TUBERS)

- **Dataset:** TUBERS ("Visual Data from Water Pipe Inspection")
- **Source:** Zenodo Record 20136698 (DOI: `10.5281/zenodo.20136698`)
- **Audit Findings:** 930 total images audited; archive hash matches expected. Cross-partition near-duplicates (Hamming $\le 4$) and capture-session prefix overlap detected across train/val/test partitions.
- **Evaluation Eligibility:** **DISABLED**. Partition leakage precludes treating this dataset as an independent benchmark for real-world generalization.
