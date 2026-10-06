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

- **OpenCV-Based Corrosion Screening:** Robust color segmentation tailored for rust and orange-brown surface staining.
- **Structural Foreground ROI Detection:** Edge detection and morphological grouping to isolate equipment boundaries.
- **Extent & Noise Filtering:** Rejection of tiny background artifacts and giant frame-spanning background contours.
- **Region Detection & Bounding Boxes:** Accurate bounding box extraction and region counting for identified corrosion clusters.
- **Deterministic Agent Decision Layer:** Rule-based classification producing CRITICAL, WARNING, or PASS operational states.
- **Human Approval Enforcement:** Mandatory supervisor review and approval before any CRITICAL maintenance work-order proposal can proceed.
- **Input Validation & Safety Layer:** Dimension sanity checks (minimum 128x128 pixels) to prevent unreliable inspection on undersized or corrupt images.
- **AWS Lambda Adapter:** Ready-to-deploy serverless adapter supporting Function URLs with structured JSON error handling (HTTP 400/200).
- **Reproducible Evaluation Harness:** Automated evaluation script (`tools/evaluate.py`) capturing environment metrics, performance, robustness, and safety evidence.
- **Comprehensive Automated Tests:** 34 robust unit, integration, robustness, and validation test cases.

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

The AWS-targeted Linux runtime and OpenCV 5 compliance are fully verified via automated GitHub Actions CI builds.

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

*Status Note:* Live AWS deployment is pending account activation; the Lambda artifact and Linux/OpenCV 5 runtime have already been successfully built and tested in CI.

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

**Test Suite Summary:** `34/34` tests passed successfully across all inspection, agent, robustness, and validation suites.

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
│   ├── main.py              # Integrated CLI workflow application
│   └── validation.py        # Input safety validation layer
├── vision/
│   └── inspection.py        # Frozen OpenCV corrosion inspection pipeline
├── tests/
│   ├── test_inspection.py   # Vision inspection & lambda adapter tests
│   ├── test_agent_decision.py # Agent unit & end-to-end tests
│   ├── test_robustness.py   # 12-category robustness & failure-case tests
│   └── test_validation.py   # Input safety validation tests
├── tools/
│   └── evaluate.py          # Reproducible evaluation harness script
├── data/
│   ├── pipe_corrosion.jpg   # Real industrial test image (1920x1080)
│   └── outputs/             # Generated evaluation reports and artifacts
├── lambda_handler.py        # AWS Lambda serverless function adapter
├── requirements.txt         # Local development dependencies
├── requirements-lambda.txt  # AWS Lambda deployment dependencies
└── README.md                # Project documentation
```

---

## Limitations

- **Scene Sensitivity:** Classical computer vision algorithms are sensitive to extreme lighting variations, reflections, and complex background clutter.
- **Geometric Filtering:** Synthetic rectangular corrosion patches can be filtered out by classical structural ROI extent criteria.
- **Screening Estimate:** Corrosion percentages represent computer-vision screening estimates rather than certified physical metallurgical measurements.
- **Human-in-the-Loop:** CRITICAL operational findings strictly require human supervisor review and approval.
- **AWS Status:** Live cloud deployment is pending account activation; runtime artifacts are fully CI-verified.

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
