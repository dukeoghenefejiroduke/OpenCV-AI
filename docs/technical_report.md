# CementSight — Technical Report

## 1. Executive Summary

CementSight is an AI-powered visual inspection and decision-support system engineered for industrial equipment screening, specifically targeting corrosion detection in cement manufacturing infrastructure. Operating in high-stress and abrasive industrial environments, equipment such as kilns, calciners, ductwork, piping, and structural supports are prone to material degradation. Left unaddressed, surface oxidation and corrosion can compromise structural integrity, leading to catastrophic equipment failure, costly unscheduled downtime, and severe safety hazards.

Traditional industrial inspection relies heavily on manual visual auditing by certified inspectors. While thorough, manual inspection is time-consuming, subjective, difficult to standardize across plants, and limited by human fatigue and accessibility constraints. CementSight addresses these challenges by combining robust, deterministic computer vision with an explicit decision-support agent and rigorous safety boundaries.

The system architecture features:
- **OpenCV-based Computer Vision:** Color-space segmentation in HSV, structural region-of-interest (ROI) isolation, and morphological filtering (`vision/inspection.py`) to quantify surface corrosion coverage and identify candidate defect regions.
- **Deterministic Agent Decision Layer:** Rule-based decision logic (`agent/decision.py`) that translates quantitative vision metrics into actionable maintenance recommendations (`PASS`, `WARNING`, `CRITICAL`) without probabilistic hallucination or LLM randomness.
- **Input Validation & Safety Boundaries:** Strict pre-inspection validation (`app/validation.py`) enforcing minimum dimension checks (128x128) and robust decodability verification to reject malformed or unsuitable inputs before they reach analytical pipelines.
- **Human-in-the-Loop Oversight:** Mandatory `approval_required=True` gating for `CRITICAL` conditions, preventing automated or unsafe external work-order dispatch.
- **Cloud-Ready AWS Lambda Architecture:** Containerized and CI-verified serverless architecture (`lambda_handler.py`, `.github/workflows/build-lambda.yml`) designed for scalable, stateless execution on Linux x86_64 with OpenCV 5.

*Current Deployment Status:* The codebase, test suite (34/34 passing), evaluation harness, and Lambda deployment packaging are fully implemented and verified in CI. Live AWS deployment is currently **PENDING ACCOUNT ACTIVATION**. CementSight is positioned as a screening and decision-support system and must not be oversold as a certified, autonomous industrial inspection authority.

---

## 2. Problem Definition

Industrial cement production involves extreme operating conditions, including high temperatures, abrasive particulate flows, chemical exposure, and moisture. Under these conditions, corrosion and metal loss represent primary failure modes.

Key challenges in industrial corrosion monitoring include:
- **Visual Subtlety and Background Complexity:** Surface oxidation is frequently mixed with rusty industrial background materials, dust, discoloration, and varying illumination. Naive color thresholding often overestimates corrosion by misclassifying legitimate rust-colored background surfaces as active defects.
- **Subjectivity and Standardization:** Manual inspections vary between inspectors, making longitudinal tracking and objective severity trending difficult.
- **Traceability and Reproducibility:** Plant operators require repeatable screening metrics and auditable decision trails to prioritize maintenance work orders efficiently.

*Scope Clarification:* CementSight is designed as an automated screening and decision-support system to assist qualified inspectors in prioritizing maintenance. It does **not** replace certified metallurgical inspection or structural engineering assessment.

---

## 3. System Objectives

CementSight was designed to meet specific, measurable engineering objectives:
1. Perform image-based corrosion screening on industrial inspection imagery.
2. Isolate structurally relevant foreground regions to suppress background rust interference.
3. Estimate surface corrosion coverage percentage accurately.
4. Identify candidate corrosion regions with bounding boxes and bounding coordinates.
5. Convert quantitative vision outputs into deterministic maintenance recommendations (`PASS`, `WARNING`, `CRITICAL`).
6. Reject clearly unsuitable or undersized input images before analysis.
7. Provide reproducible CLI (`app/main.py`), programmatic API, and AWS Lambda behavior.
8. Verify OpenCV 5 compatibility in the competition and cloud target environment.
9. Preserve mandatory human approval for critical maintenance actions.

---

## 4. System Architecture

The CementSight architecture is documented in `docs/architecture.md` and illustrated in `docs/architecture.svg`. 

```
[Input Image (CLI / Base64 / Lambda Event)]
                   │
                   ▼
       [Input Validation Layer] (`app/validation.py`)
       - Readability & Decodability Check
       - Minimum Dimension Check (>= 128x128)
                   │
                   ▼ (If Valid)
       [OpenCV Vision Pipeline] (`vision/inspection.py`)
       - HSV Conversion & Color Segmentation
       - Structural ROI Masking & Morphology (7x7 Kernel)
       - Corrosion Coverage % & Bounding Regions
                   │
                   ▼
       [Agent Decision Layer] (`agent/decision.py`)
       - Deterministic Rule Evaluation (PASS / WARNING / CRITICAL)
       - Human Approval Gate (approval_required for CRITICAL)
                   │
                   ▼
       [Output / Evidence Path] (CLI stdout / JSON Report / Lambda Response)
```

### Environment Separation
CementSight maintains strict awareness of its operating environments:
- **Local Development Environment:** Android / Termux / ARM64 / Python 3.14.6 / OpenCV 4.14.0. (Used for local iteration, script execution, and test execution).
- **Competition and Cloud Target Environment:** Linux x86_64 / Python 3.11 / NumPy 2.2.3 / OpenCV 5.0.0.93. (Built and verified automatically via GitHub Actions CI).

---

## 5. Computer Vision Pipeline

The core computer vision logic in `vision/inspection.py` executes an eleven-stage pipeline designed for industrial pipe and equipment inspection:

1. **Gaussian Blur:** Reduces high-frequency sensor noise and micro-texture variations.
2. **HSV Conversion:** Converts BGR images to the Hue-Saturation-Value (HSV) color space for robust chromaticity isolation.
3. **Corrosion-Color Segmentation:** Applies tuned HSV lower and upper bounds corresponding to rust and iron oxide signatures.
4. **Structural Foreground Extraction:** Analyzes grayscale luminance and edge distribution to isolate the primary equipment structure from the background.
5. **Contour/Extent Filtering:** Filters extracted contours based on geometric extent and area ratios.
6. **ROI Restriction:** Restricts inspection to valid structural regions-of-interest.
7. **Morphology:** Applies morphological opening and closing operations using a frozen **7x7 kernel** to eliminate noise and bridge fragmented corrosion patches.
8. **Corrosion Coverage Calculation:** Computes the ratio of active corrosion pixels to total structural foreground pixels.
9. **Connected-Region Extraction:** Identifies discrete connected components corresponding to corrosion clusters.
10. **Bounding-Box Generation:** Calculates bounding rectangles (`[x, y, w, h]`) for each candidate region.
11. **Recommendation Generation:** Maps coverage percentage to preliminary recommendation categories (`PASS`, `WARNING`, `CRITICAL`).

### Importance of Structural ROI Filtering
In real industrial imagery (e.g., `data/pipe_corrosion.jpg`), background surfaces frequently exhibit residual rust stains or discoloration. Without structural foreground extraction and ROI restriction, naive color thresholding overestimates corrosion coverage. Restricting analysis to the primary equipment structure ensures that background degradation artifacts do not distort the screening score.

*Limitations:* The algorithm is a classical computer vision pipeline and is not universally accurate across all possible lighting, viewing angles, or exotic alloy types.

---

## 6. Input Validation & Safety Boundary

Input validation is implemented in `app/validation.py` and executes **before** any vision inspection or agent processing occurs.

### Validation Checks
- **Readability / Decodability:** Verifies that image files exist, can be opened, and decode successfully into valid NumPy arrays via OpenCV (`cv2.imdecode`).
- **Dimension Validation:** Enforces minimum dimension thresholds.
- **Minimum Resolution Threshold (128x128):** 
  - *Engineering Rationale:* Initial testing demonstrated that excessively small images (e.g., 32x32 pixels) produced unreliable, distorted results (such as an artifactual 34.86% CRITICAL score on tiny inputs). Real industrial inspection images (e.g., `data/pipe_corrosion.jpg` at 1920x1080) and test suites are significantly larger. 
  - The 128x128 threshold acts as a defensible engineering guardrail that rejects undersized or thumbnail images as `INPUT_REJECTED`. 
  - *Caveat:* Input validation ensures data integrity and prevents pipeline crashes, but it does **not** guarantee inspection accuracy.

---

## 7. Agentic Vision Decision Layer

The agentic decision layer (`agent/decision.py`) translates numerical corrosion percentages and vision recommendations into deterministic maintenance actions without LLM inference or stochastic variability.

### Deterministic Rules
- **PASS:** `corrosion < 2.0%`
- **WARNING:** `2.0% <= corrosion < 10.0%`
- **CRITICAL:** `corrosion >= 10.0%` or when upstream inspection recommendation is `CRITICAL`.

### Agent Characteristics
- **No LLM Required:** Executes via pure, deterministic Python logic.
- **Direct Vision Influence:** Vision outputs directly drive action selection.
- **Proposed Actions:** Tailored maintenance recommendations are generated based on severity.
- **Human Approval Gate:** `approval_required = True` is strictly enforced for all `CRITICAL` decisions.
- **No Automatic Dispatch:** The system generates decision records and recommendations but does **not** autonomously dispatch external work orders or initiate physical plant actions.
- **Conservative Error Handling:** Malformed or incomplete vision data is handled conservatively, defaulting to safe fallback states.

---

## 8. Lambda/API Adapter

The serverless adapter (`lambda_handler.py`) exposes CementSight to cloud environments via AWS Lambda event payloads.

### Request/Response Lifecycle
- **Input Forms:** Accepts JSON payloads containing base64-encoded image data (`image_base64`) or direct data URLs.
- **Pipeline Execution:** Decodes the base64 payload, passes the image through input validation (`app/validation.py`), executes the vision inspection (`vision/inspection.py`), and invokes the agent decision layer (`agent/decision.py`).
- **Structured JSON Response:** Returns comprehensive inspection and decision metadata.
- **HTTP Status Codes:**
  - `HTTP 200`: Successful inspection and decision generation.
  - `HTTP 400`: Invalid client input, decoding failure, or validation rejection (e.g., undersized 32x32 image).
  - `HTTP 500`: Unexpected runtime errors.

*Deployment Note:* While containerized and verified in CI, there is no externally accessible live AWS Lambda endpoint currently active.

---

## 9. OpenCV 5 Compliance

OpenCV 5 compliance is a key architectural pillar of the competition target.

### Environment Divergence & Justification
- **Local Development (Android / Termux / ARM64 / Python 3.14.6):** Operates on OpenCV 4.14.0 because precompiled OpenCV 5 wheels are incompatible with Android ARM64 under Python 3.14. This is an environment constraint, not an avoidance of OpenCV 5.
- **Competition & Cloud Target (Linux x86_64 / Python 3.11):** Built and verified via GitHub Actions CI (`.github/workflows/build-lambda.yml`) using OpenCV 5.0.0.93 and NumPy 2.2.3.

### Measured Lambda Package Metrics
- **Compressed ZIP Size:** `74.10 MB`
- **Uncompressed Package Size:** `207.86 MB`
- **AWS Lambda Uncompressed Limit:** `250.00 MB`
- *Status:* Comfortably within AWS Lambda limits; import verification and handler tests against OpenCV 5 passed successfully in CI.

---

## 10. Evaluation Methodology

The evaluation harness (`tools/evaluate.py`) assesses system behavior across a broad spectrum of synthetic and real scenarios.

### Tested Categories
1. **Real Industrial Image:** `data/pipe_corrosion.jpg` (1920x1080).
2. **Clean Synthetic Image:** Synthetic undamaged pipe surface.
3. **Moderate Synthetic Corrosion:** Synthetic rust simulation (expected `WARNING`).
4. **High Synthetic Corrosion Scenario:** High corrosion simulation.
5. **Dark Image:** Low-illumination robustness test.
6. **Bright Image:** Overexposed illumination test.
7. **Noisy Image:** High-noise sensor test.
8. **Tiny 32x32 Image:** Undersized input rejection test.
9. **Invalid Input:** Corrupt or missing file path handling.
10. **Agent Critical Safety Case:** Controlled 18.5% corrosion scenario verifying mandatory human approval gating.

*Scope Note:* These scenarios represent engineering robustness and regression verification tests, **not** a statistically representative industrial benchmark dataset.

---

## 11. Results

### Real Industrial Image Evaluation (`data/pipe_corrosion.jpg`)
- **Dimensions:** `1920x1080`
- **Corrosion Estimate:** `6.56%`
- **Detected Regions:** `11`
- **Vision Recommendation:** `WARNING`
- **Agent Decision:** `WARNING`
- **Approval Required:** `False`
- **Proposed Action:** `Proposed scheduling a routine maintenance inspection and ongoing monitoring.`

### Test Suite Execution Summary
All tests executed via `PYTHONPATH=. python3 tests/...` pass successfully:

| Test Script | Passing Tests | Status |
| :--- | :---: | :---: |
| `tests/test_inspection.py` | 6 / 6 | PASSED |
| `tests/test_agent_decision.py` | 9 / 9 | PASSED |
| `tests/test_robustness.py` | 12 / 12 | PASSED |
| `tests/test_validation.py` | 7 / 7 | PASSED |
| **Total Test Suite** | **34 / 34** | **PASSED** |

### Key Behavioral Findings
- Tiny `32x32` images are correctly rejected as `INPUT_REJECTED / UNSUITABLE`.
- Controlled critical agent tests (`18.5%` corrosion) correctly trigger `approval_required=True`.
- No false external work-order dispatch claims are made by the agent.

---

## 12. Robustness & Failure Cases

To maintain scientific integrity, known failure modes and limitations are documented openly rather than concealed:
- **Background Interference:** Strong rust-colored background textures can occasionally challenge color segmentation, requiring structural ROI filtering.
- **Scene Sensitivity:** Structural ROI assumptions depend on visible equipment geometry and edge contrast.
- **Synthetic Geometry Rejection:** Certain high-corrosion synthetic rectangles can be rejected by structural ROI extent filtering in specific test configurations.
- **Screening Metric vs. Physical Measurement:** The corrosion percentage is a computer vision screening estimate, not a certified physical metal-loss measurement.
- **Bounding Boxes:** Bounding boxes identify candidate regions of interest for inspector review, not certified defect dimensions.
- **Input Constraints:** Undersized, dark, bright, and noisy inputs are handled conservatively by validation and vision pipelines.

---

## 13. Safety & Responsible Operation

CementSight incorporates multiple layers of safety and responsible design:
- **Pre-Inspection Validation:** Invalid or malformed inputs are rejected before analytical execution.
- **Deterministic Rules:** Decisions are transparent and reproducible, eliminating black-box stochastic hallucinations.
- **Conservative Fallbacks:** Malformed data or pipeline errors trigger safe fallback states.
- **Mandatory Human Approval:** `CRITICAL` conditions require explicit human supervisor sign-off (`approval_required=True`).
- **No Autonomous Dispatch:** The system acts strictly as a decision-support screener; it never dispatches external work orders automatically.
- **Transparent Limitations:** All operational limitations are explicitly documented.

---

## 14. Reproducibility

All pipeline operations, evaluations, and tests are fully reproducible using standard commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual test suites
PYTHONPATH=. python3 tests/test_inspection.py
PYTHONPATH=. python3 tests/test_agent_decision.py
PYTHONPATH=. python3 tests/test_robustness.py
PYTHONPATH=. python3 tests/test_validation.py

# Run evaluation harness
PYTHONPATH=. python3 tools/evaluate.py

# Run CLI inspection on real industrial image
PYTHONPATH=. python3 app/main.py data/pipe_corrosion.jpg
```

### GitHub Actions CI & Lambda Build
The workflow `.github/workflows/build-lambda.yml` automatically provisions Python 3.11 on Linux x86_64, installs pinned dependencies (`requirements-lambda.txt`), packages the Lambda deployment artifact, verifies OpenCV 5 (`cv2.__version__ == '5.0.0.93'`) and NumPy 2 (`np.__version__ >= '2.0'`), executes the Lambda handler against `data/pipe_corrosion.jpg`, and runs the test suite.

---

## 15. CI/CD and Cloud Delivery

Continuous integration is managed via GitHub Actions (`.github/workflows/build-lambda.yml`), ensuring cross-platform build stability and artifact compliance.

### Pipeline Steps
1. Checkout repository.
2. Setup Python 3.11 environment on `ubuntu-latest`.
3. Install build and test dependencies (`requirements.txt`).
4. Run core unit test suite.
5. Build Lambda deployment package (`lambda_build/`) using binary-only wheels (`requirements-lambda.txt`).
6. Verify package uncompressed size (`207.86 MB` < `250 MB` limit).
7. Execute runtime import verification for OpenCV 5.0.0.93 and NumPy 2.2.3.
8. Execute Lambda handler integration test using `data/pipe_corrosion.jpg`.
9. Generate deployment artifact (`cementsight_lambda.zip`, `74.10 MB`).

*Cloud Deployment Status:* **AWS LIVE DEPLOYMENT: PENDING ACCOUNT ACTIVATION.**

---

## 16. Competition Requirements Mapping

| Requirement | CementSight Evidence | Status |
| :--- | :--- | :--- |
| **OpenCV 5 Substantive Image Analysis** | `vision/inspection.py`, CI OpenCV 5 verification | Verified (CI Cloud Target) / 4.14 (Local) |
| **AWS Component** | `lambda_handler.py`, `.github/workflows/build-lambda.yml` | Implemented & CI-Verified (Live Deployment Pending) |
| **Reproducible Code** | `app/main.py`, `tools/evaluate.py`, tests | Fully Verified (34/34 Passing) |
| **Dependency Pinning** | `requirements.txt`, `requirements-lambda.txt` | Fully Pin-Verified |
| **Test Suite** | `tests/` (4 test files) | 34 Passed, 0 Failed |
| **Evaluation Evidence** | `tools/evaluate.py`, `data/outputs/evaluation_report.json` | Fully Generated |
| **Architecture & Documentation** | `docs/architecture.md`, `docs/technical_report.md` | Complete |
| **Safety & Human Oversight** | `agent/decision.py`, `approval_required=True` | Implemented |
| **Demo Readiness** | CLI, Lambda adapter, evaluation harness | Ready |

---

## 17. Innovation

CementSight introduces several practical innovations for industrial computer vision:
- **Structural ROI Restriction:** Effectively suppresses background rust contamination in complex industrial scenes.
- **Deterministic Vision-to-Action Pipeline:** Bridges low-level image processing with high-level maintenance recommendations without probabilistic LLM hallucination.
- **Explicit Human Approval Gate:** Enforces safety governance for critical degradation findings.
- **Pre-Inspection Validation:** Rejects unsuitable inputs before analytical execution.
- **Cloud-Ready OpenCV 5 Packaging:** Demonstrates clean packaging within AWS Lambda size limits.
- **Reproducible Evaluation Harness:** Provides automated reporting for robustness and regression scenarios.

---

## 18. Limitations

To ensure absolute technical honesty, the following limitations apply:
- **Scene Sensitivity:** Performance varies with lighting, viewing angle, and surface condition.
- **Classical Segmentation:** Color thresholding is sensitive to extreme illumination changes.
- **Structural ROI Assumptions:** Assumes identifiable structural geometry.
- **Lack of Large Dataset:** Evaluated on representative industrial and synthetic samples rather than large-scale labeled industrial datasets.
- **Screening Estimate:** Corrosion percentage is an image-space screening estimate, not certified physical thickness loss.
- **AWS Deployment Status:** Live cloud endpoint is pending account activation.
- **Non-Certified System:** CementSight is a decision-support aid, not a certified maintenance authority.

---

## 19. Future Work

Potential avenues for future development include:
- Integration of larger labeled industrial corrosion datasets.
- Quantitative ground-truth calibration against physical ultrasonic thickness measurements.
- Expansion to additional industrial defect types (e.g., cracking, spalling, coating blistering).
- Temporal video inspection for longitudinal degradation tracking.
- Cloud deployment upon AWS account activation.
- Enterprise API authentication and integration with Computerized Maintenance Management Systems (CMMS).

---

## 20. Conclusion

CementSight has successfully delivered a robust, reproducible, and safety-conscious industrial vision inspection and decision-support prototype. 

Key demonstrated achievements include:
- A fully functional OpenCV vision inspection pipeline (`vision/inspection.py`).
- A deterministic agent decision layer with mandatory human oversight (`agent/decision.py`).
- Comprehensive pre-inspection input validation (`app/validation.py`).
- A fully passing test suite (34/34 tests passing).
- OpenCV 5 cloud-target verification via GitHub Actions CI.
- Complete architectural documentation and reproducible evaluation harness.
- *AWS live deployment pending account activation.*

CementSight provides a solid technical foundation for automated industrial equipment screening and maintenance decision support.
