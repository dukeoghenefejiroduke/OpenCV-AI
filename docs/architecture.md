# CementSight Architecture & System Design

CementSight is an industrial visual inspection system designed to analyze imagery of cement plant equipment and industrial pipes, detect corrosion and surface degradation through classical computer vision (OpenCV), and translate findings into traceable operational decisions using a deterministic agentic decision layer.

---

## 1. Architecture Overview

CementSight adopts a modular, deterministic, and security-first architecture. By decoupling the classical computer vision inspection engine from the agentic decision layer and the application deployment boundary, the system ensures complete reproducibility, zero LLM hallucination risks, and robust local-to-cloud portability.

```
[User / Client] 
      │
      ▼
[Input Validation Safety Layer] (≥ 128x128 pixels, format check)
      │
      ▼
[OpenCV Vision Pipeline] (vision/inspection.py — Frozen)
      │
      ▼
[Deterministic Agent Decision Layer] (agent/decision.py)
      │
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
[PASS (< 2%)]                 [WARNING (2%–<10%)]          [CRITICAL (≥ 10%)]
Normal Status Recorded        Routine Maintenance Proposal    Human Approval Required
                                                              (No Automatic Dispatch)
```

---

## 2. Component Responsibilities

1. **Input Validation Layer (`app/validation.py`):**
   - Intercepts raw image paths or numpy arrays at the application boundary.
   - Verifies file existence, decodability, and non-empty shape.
   - Enforces a defensible minimum dimension threshold (`128x128` pixels) to prevent unreliable inspection on undersized or corrupt images.

2. **OpenCV Vision Pipeline (`vision/inspection.py` — Frozen):**
   - Applies Gaussian blur to suppress high-frequency noise.
   - Converts frames to HSV color space for robust rust/orange-brown color segmentation (`lower_rust=[5,50,50]`, `upper_rust=[25,255,255]`).
   - Computes classical structural foreground Region of Interest (ROI) masks via Canny edge detection, morphological closing/dilating, and extent filtering (`extent ≥ 0.15`, rejecting giant frame-spanning contours).
   - Performs contour analysis and morphological cleaning to extract corrosion percentage, region count, and bounding boxes.

3. **Deterministic Agent Decision Layer (`agent/decision.py`):**
   - Consumes structured inspection output dictionaries.
   - Applies strict, rule-based thresholds (`PASS < 2.0%`, `WARNING 2.0%–<10.0%`, `CRITICAL ≥ 10.0%`).
   - Formulates operational decisions, proposed maintenance actions, approval requirements (`approval_required = True` for CRITICAL), and transparent reasoning.
   - Enforces strict safety constraints prohibiting any false claims of automated external work-order dispatch.

4. **Application & Serverless Adapters (`app/main.py` & `lambda_handler.py`):**
   - Provides a clean CLI interface for local execution and demonstration.
   - Provides an AWS Lambda serverless adapter supporting Function URLs with structured JSON responses (HTTP 200/400).

---

## 3. End-to-End Data Flow & Evidence Path

```
Image Input 
    │
    ▼
Input Validation (app/validation.py)
    │
    ▼
OpenCV Inspection (vision/inspection.py)
    ├─► Corrosion Percentage (%)
    ├─► Region Count & Bounding Boxes
    └─► Vision Recommendation (PASS / WARNING / CRITICAL)
    │
    ▼
Agent Decision Evaluation (agent/decision.py)
    ├─► Decision State (PASS / WARNING / CRITICAL)
    ├─► Proposed Action (Routine inspection / Work-order request)
    ├─► Approval Required (False / True)
    └─► Detailed Reasoning
    │
    ▼
Structured JSON / CLI / API Response
    │
    ▼
Evaluation & Test Evidence (tools/evaluate.py & 34 automated tests)
```

---

## 4. Environment Matrix: Local vs. CI vs. AWS Target

CementSight employs a dual-environment strategy to accommodate local ARM64 constraints while maintaining cutting-edge cloud compliance:

| Dimension | Local Development | GitHub Actions CI | Competition / AWS Target |
| :--- | :--- | :--- | :--- |
| **Operating System** | Android / Termux (ARM64) | Linux x86_64 Runner | AWS Lambda (Amazon Linux) |
| **Python Version** | Python 3.14.6 | Python 3.11 | Python 3.11 |
| **NumPy Version** | NumPy 2.x | NumPy 2.2.3 | NumPy 2.2.3 |
| **OpenCV Version** | OpenCV 4.14.0 (compatible local wheel) | OpenCV 5.0.0.93 | OpenCV 5.0.0.93 (`opencv-python-headless`) |
| **Deployment Status** | Active (Local execution & testing) | Verified (Automated build & test) | **Pending Account Activation** |

---

## 5. OpenCV 5 Compliance Explanation

OpenCV 5 (`opencv-python-headless==5.0.0.93`) is the designated cloud and competition target runtime, providing high-performance headless execution. However, precompiled OpenCV 5 binaries are not compatible with local Android ARM64 / Python 3.14 environments. 

To resolve this without compromising local developer velocity or cloud compliance:
1. **Shared Core:** The vision inspection algorithm (`vision/inspection.py`) is 100% shared and frozen between both environments.
2. **Local Fallback:** Local development utilizes OpenCV 4.x compatible wheels.
3. **CI Verification:** GitHub Actions automatically builds and tests the exact AWS-targeted Linux x86_64 / Python 3.11 / OpenCV 5 environment.

---

## 6. Agent Safety Model

The agentic decision layer enforces strict safety invariants:
- **No LLMs / Randomness:** Purely deterministic threshold rules.
- **Human Oversight:** CRITICAL states explicitly mandate human supervisor review (`approval_required = True`).
- **No Automated Dispatch:** Actions are strictly formulated as *proposals/requests* (`"Proposed immediate emergency maintenance work-order request. Awaiting supervisor approval."`). The system never claims to dispatch or execute external work orders.
- **Defensive Validation:** Malformed inspection reports are intercepted and handled conservatively.

---

## 7. Current AWS Deployment Status

- **Build Status:** Successfully compiled and packaged via GitHub Actions.
- **Package Size:** ~74.10 MB compressed (ZIP) / ~207.86 MB uncompressed.
- **Runtime Tests:** OpenCV 5 runtime and Lambda handler tests passed successfully in CI.
- **Live Status:** **Live AWS deployment is pending account activation.** No live Lambda Function URLs are currently deployed or claimed.

---

## 8. Reproducibility & Testing

CementSight maintains 100% test coverage across 34 automated tests:
- `tests/test_inspection.py` (Vision & Lambda adapter tests)
- `tests/test_agent_decision.py` (Agent decision unit & E2E tests)
- `tests/test_robustness.py` (12-category robustness & failure cases)
- `tests/test_validation.py` (Input safety validation tests)
- `tools/evaluate.py` (Reproducible evaluation harness producing `data/outputs/evaluation_report.json`)
