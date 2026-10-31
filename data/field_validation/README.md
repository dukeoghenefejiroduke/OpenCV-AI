# CementSight AI Field Validation Dataset & Annotation Framework (v1.3.0 / v1.4.0)

## Overview

This directory establishes a lightweight, reproducible dataset manifest and human-annotation framework for evaluating CementSight AI across its five core hazard screening domains:
1. **Corrosion and Rust Inspection** (`corrosion`)
2. **Structural Crack Screening** (`crack`)
3. **Surface Defect Screening** (blistering, spalling) (`surface_defect`)
4. **Leakage and Seepage Indicator Screening** (`leakage`)
5. **Insulation Damage and Cladding Screening** (`insulation`)

Plus explicit **Clean / Negative Examples** (`clean_negative`) containing common confounders (welds, brackets, shadows, grime, ordinary joints, and paint variations).

---

## Dataset Structure & Provenance Guidelines

- **Provenance**: Each record must explicitly declare whether the image is `real`, `synthetic`, or `generated`. Synthetic and generated images are maintained in separate fixture tests or development partitions and are **strictly segregated** from real-world performance claims.
- **License & Attribution**: Real-world records must document their original source, source URL, license, attribution requirements, retrieval date, and usage permissions. Unverified or restricted sources are excluded from evaluation partitions.
- **Privacy**: No sensitive facility information, personal identifiers, API keys, or credentials are included in datasets or metadata.
- **Splits & Grouping**: Records are assigned to dataset splits (`development`, `validation`, `held_out_test`). A `grouping_id` is mandatory for related images or bursts of the same asset to prevent split leakage across development, validation, and held-out test partitions.

---

## Human-Review Process

To ensure high-data quality before any image becomes eligible for evaluation, annotators and reviewers follow a 5-step protocol:
1. **Source & Permission Verification**: Verify the source and permitted license/usage terms.
2. **Hazard Relevance**: Confirm that the image genuinely depicts the stated hazard.
3. **Evidence-Based Labeling**: Assign a label (`positive`, `negative`, `ambiguous`) only when supported by visible evidence.
4. **Uncertainty & Ambiguity**: Record `"ambiguous"` or `"insufficient_evidence"` rather than guessing. Never force uncertain imagery into positive or negative labels. Unreviewed records (`annotation_status: "unreviewed"`) are **never** treated as confirmed negatives.
5. **Bounding Box Adjudication**: Record bounding boxes only where meaningful and supported by visible evidence, validated against image dimensions.

---

## Critical Domain Limitations

Visual computer-vision screening has inherent boundaries that must be respected during evaluation:
- **Leakage**: RGB imagery alone cannot confirm an active leak or seepage in every case (distinguishing condensation, stains, and active weeping requires corroborating evidence).
- **Insulation**: Surface cladding appearance alone cannot establish internal insulation thermal performance or hidden core degradation.
- **Structural Integrity**: Surface appearance alone cannot establish internal structural load capacity or subsurface cracking.
- **Physical Measurements**: Pixel measurements and bounding box areas are strictly optical screening estimates and **never** report physical units (e.g. millimeters) without a validated scale reference and calibration.

---

## Dataset Validation Utility

Run the dataset validator to check manifest integrity, missing files, out-of-bounds bounding boxes, duplicate IDs, split leakage, and annotation states:

```bash
PYTHONPATH=. python tools/validate_dataset.py
```
