# CementSight AI - Field Demonstration Script & Rehearsal Guide

## Demo Overview
This script guides operators and reviewers through testing 4 distinct scenarios using the CementSight AI local dashboard (`tools/server.py`).

---

## 1. Demo Scenarios & Expected Outputs (Real Run Baselines)

### Scenario A: Clean Steel Pipe (`bare_or_painted_metal`)
- **Input Image:** Clean metal pipe (`data/pipe_corrosion.jpg` or clean asset).
- **Surface Selection:** `Bare / Painted Metal`
- **Expected Output:**
  - Corrosion Estimate: `0.0%` (or low background noise within tolerance).
  - Verdict: `PASS` / `Routine monitoring`.
  - Recommendation: Routine maintenance inspection & monitoring.

### Scenario B: Rusty Steel Pipe (`bare_or_painted_metal`)
- **Input Image:** Heavily corroded industrial pipe (`data/pipe_corrosion.jpg`).
- **Surface Selection:** `Bare / Painted Metal`
- **Expected Output:**
  - Corrosion Estimate: `6.56%` (11 regions identified).
  - Verdict: `WARNING`
  - Recommendation: Schedule localized non-destructive testing (UT) and touch-up coating.

### Scenario C: Wood / Non-Metal Surface Selected (`brick_concrete_wood`)
- **Input Image:** Wooden plank or red brick surface.
- **Surface Selection:** `Brick / Concrete / Wood`
- **Expected Output:**
  - Corrosion Estimate: `0.0%` (Screening bypassed).
  - Verdict / Notice: `Corrosion screening is not applicable to this surface.`
  - Findings: None (preventing false-positive alarms on wood/brick).

### Scenario D: Uncertain Surface Selected (`not_sure`)
- **Input Image:** Rust-coloured wooden or textured surface.
- **Surface Selection:** `Not Sure / Other`
- **Expected Output:**
  - Screening executed with prominent warning: `WARNING: Rust-coloured materials (brick, wood, concrete) can trigger false alarms.`
  - Review state and supervisor approval required.

### Scenario E: Non-Steel Metal Surface Selected (`other_metal_non_steel`)
- **Input Image:** Copper pipe, brass fitting, or aluminium structure.
- **Surface Selection:** `Other Metal (Copper, Brass, Aluminium)`
- **Expected Output:**
  - Corrosion Estimate: `0.0%` (Screening bypassed).
  - Verdict / Notice: `Corrosion screening is not applicable to this surface (copper, brass, aluminium or other non-steel metal).`
  - Findings: None (bypassing steel corrosion rules for non-steel metals).

---

## 2. Rehearsal Checklist & Hard Negatives Setup
- **Hard Negatives Testing:** Test against known tricky subjects (outside repo / lockbox photos): copper pipes, mustard or yellow insulated pipes, hazy and steamy inspection scenes, and orange lighting reflecting on clean metal.
- **Lighting:** Ensure ambient illumination is even; avoid direct flashlight glare or intense specular reflections on metallic surfaces which can saturate HSV color channels.
- **Framing & Distance:** Maintain consistent distance (approx. 0.5m - 1.5m) perpendicular to the inspection surface.
- **Offline Readiness:** Verify local server runs successfully via `python tools/server.py` and is accessible at `http://localhost:8000`.
- **Browser Security:** Ensure browser permissions allow camera or file upload in local offline mode.
