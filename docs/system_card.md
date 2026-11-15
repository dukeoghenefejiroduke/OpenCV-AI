# CementSight AI — System Card

## 1. Intended Use
CementSight AI is an edge-first computer vision and operational intelligence screening tool designed for industrial equipment, pipes, and cement plant infrastructure inspection. It provides automated preliminary surface condition screening (corrosion percentage estimation, multi-hazard detection, and heuristic risk prioritization) to assist certified human inspectors and maintenance teams.

## 2. Out-of-Scope Surfaces & Hard Negatives
- **Non-Steel Metals (`other_metal_non_steel`):** Copper pipes, brass fittings, and aluminium structures are explicitly out of scope for steel corrosion screening. Selecting this surface bypasses screening and reports "Corrosion screening is not applicable to this surface (copper, brass, aluminium or other non-steel metal)".
- **Non-Metallic Textured Surfaces:** Surfaces such as brown wood (`wood` category) and red brick (`red_brick` category) exhibit warm color distributions that overlap with HSV corrosion thresholds.
- **Hard Negatives & Complex Environmental Conditions:** Copper/brass pipes, mustard or yellow insulated pipes, hazy and steamy scenes, and orange lighting on clean metal can produce chromatic false-positives and require careful human review with appropriate surface scoping.
- **Low Illumination & Shadows:** High-contrast shadows can suppress saturation, causing false negatives.
- **Structural Integrity:** Optical screening cannot measure internal wall thickness, stress corrosion cracking depth, pressure containment capacity, or remaining pipe lifespan.

## 3. Empirical Failure Modes & Evaluation Summary (Challenge Pack)
Evaluated against 24 challenge pack images (`data/outputs/challenge_report_20261115_171520.json`). **Note:** 22 of the 24 photos were previously used for development or earlier testing; therefore, these results serve as a regression check for known failure modes, not an accuracy or generalization benchmark.

### Detector Sensitivity & Specificity Trade-off
| Detector | Sensitivity (Rust Photos Flagged, n=12) | Specificity (Clean Photos Passed, n=12) | Overall Correct (n=24) |
| :--- | :---: | :---: | :---: |
| **Default Detector** | 0.42 (5/12) | 0.58 (7/12) | 12 / 24 (50%) |
| **Rust Detector (`RustDetector`)** | 0.75 (9/12) | 0.33 (4/12) | 13 / 24 (54%) |

*Trade-off analysis:* While the experimental rust detector increases sensitivity on corroded samples (9/12 vs 5/12), it suffers from lower specificity (falsely flagging 8 out of 12 clean photos, including silver, orange, and red brick), illustrating the inherent trade-off between missing real corrosion and generating excessive false alarms.

## 4. Mandatory Human Review Requirement
**All machine outputs, detector scores, and preliminary screening percentages are advisory only.** They do not constitute certification, engineering approval, or authorization to operate. Every inspection report requires mandatory verification and sign-off by a qualified human supervisor.
