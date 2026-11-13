# Field Corrosion Labeling Guide & Adjudication Protocol

## 1. Overview & Purpose
This guide establishes standardized data labeling and adjudication protocols for field corrosion images in CementSight AI. It ensures consistency across independent labelers and aligns directly with the schema defined in `tools/field_validation_manifest.py`.

## 2. Label Definitions for Independent Labelers (Label A & Label B)
Each field image must be independently evaluated by two trained inspectors (Labeler A and Labeler B) without sharing intermediate notes.

- **`positive` (Corrosion Present):** Visible active rust scaling, pitting, or wall thinning.
- **`negative` (Clean / Non-Corroded):** Intact coating, clean metal surface, or minor surface dust without active pitting or wall thinning.
- **`uncertain` (Borderline / Unclear):** Ambiguous surface condition requiring third-party adjudication.
- **`unusable` (Corrupt / Low Quality):** Severe blur, extreme glare, or obstructed view preventing assessment.

## 3. Borderline Photos & Adjudication Protocol
- **Borderline / Disagreement Cases:** When Label A and Label B disagree or mark `uncertain`, route the record to a designated Lead Inspector (Adjudicator). The Adjudicator reviews image metadata and assigns a definitive value in the `adjudicated_label` column. The evaluation runner strictly requires an `adjudicated_label` whenever `label_a != label_b`.

## 4. Photo-Taking Checklist (Field Inspectors)
Every field photograph must satisfy:
- **Distance:** Captured between 0.5 and 1.5 meters from the inspection target.
- **Lighting:** Uniform ambient lighting or auxiliary flash; no severe direct glare.
- **Angle:** Orthogonal ($\approx 90^\circ$) to the target surface (maximum deviation $\pm 15^\circ$).

## 5. Pre-Declared Acceptance Targets
*(To be filled in prior to evaluating any unseen field test dataset)*

- **Minimum Sensitivity (True Positive Rate):** [ ]
- **Minimum Specificity (True Negative Rate):** [ ]
- **Maximum False Alarm Bound:** [ ]
- **Detection Delay Metric:** [ ]

### Sign-off
- **Lead Quality Engineer:** ___________________________ (Date: ______________)
- **Chief Technical Officer:** ___________________________ (Date: ______________)
