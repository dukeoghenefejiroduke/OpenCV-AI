"""Cross-Hazard Benchmark and Synthetic Fixture Evaluation (tests/test_cross_hazard_benchmark.py).

Evaluates all five hazard detectors across controlled synthetic positive and negative fixtures,
reporting per-hazard metrics including True Positives (TP), False Positives (FP), True Negatives (TN),
False Negatives (FN), Precision, Recall, False-Positive Rate (FPR), and False-Negative Rate (FNR).
Clearly distinguishes synthetic-fixture evaluation metrics from real-world field performance
and explicitly avoids claiming validated real-world accuracy without representative labeled datasets.
"""

import pytest
import numpy as np
import cv2
from typing import Dict, Any, List

from vision.orchestrator import MultiHazardOrchestrator
from vision.contract import STATUS_CANDIDATE_DETECTED, STATUS_NO_CANDIDATE


def evaluate_hazard_domain(domain_name: str, positive_image: np.ndarray, negative_image: np.ndarray, detector_id: str) -> Dict[str, Any]:
    """Computes benchmark performance metrics for a specific hazard domain using positive and negative fixtures."""
    orchestrator = MultiHazardOrchestrator()

    # Evaluate positive fixture
    rep_pos = orchestrator.run_inspections(positive_image)
    status_pos = rep_pos["detectors"].get(detector_id, {}).get("execution_status")
    tp = 1 if status_pos == STATUS_CANDIDATE_DETECTED else 0
    fn = 0 if tp == 1 else 1

    # Evaluate negative fixture (clean surface)
    rep_neg = orchestrator.run_inspections(negative_image)
    status_neg = rep_neg["detectors"].get(detector_id, {}).get("execution_status")
    tn = 1 if status_neg == STATUS_NO_CANDIDATE else 0
    fp = 0 if tn == 1 else 1

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = float(fn) / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "domain": domain_name,
        "detector_id": detector_id,
        "positive_fixtures": 1,
        "negative_fixtures": 1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_positive_rate": round(fpr, 3),
        "false_negative_rate": round(fnr, 3),
        "note": "Metrics derived from controlled synthetic test fixtures; not representative of real-world field accuracy."
    }


def test_cross_hazard_per_domain_benchmark():
    """Executes per-hazard benchmark across all five hazard domains and verifies metrics reporting."""
    clean = np.full((300, 300, 3), 200, dtype=np.uint8)

    # 1. Corrosion
    corr_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.rectangle(corr_img, (100, 100), (200, 200), (30, 120, 200), -1)
    corr_metrics = evaluate_hazard_domain("corrosion", corr_img, clean, "corrosion_detector")

    # 2. Cracks
    crack_img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.line(crack_img, (50, 50), (250, 250), (20, 20, 20), 2)
    crack_metrics = evaluate_hazard_domain("crack", crack_img, clean, "crack_screening_detector")

    # 3. Surface Defects (Blistering & Spalling)
    blister_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(blister_img, (150, 150), 25, (120, 120, 120), -1)
    surface_metrics = evaluate_hazard_domain("surface_defect", blister_img, clean, "surface_defect_detector")

    # 4. Leakage & Seepage
    leak_img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(leak_img, (140, 50), (160, 250), (70, 70, 70), -1)
    leakage_metrics = evaluate_hazard_domain("leakage", leak_img, clean, "leakage_detector")

    # 5. Insulation Damage & Cladding
    insul_img = np.full((300, 300, 3), 210, dtype=np.uint8)
    cv2.rectangle(insul_img, (100, 100), (200, 200), (50, 50, 50), 3)
    insulation_metrics = evaluate_hazard_domain("insulation", insul_img, clean, "insulation_detector")

    domains = [corr_metrics, crack_metrics, surface_metrics, leakage_metrics, insulation_metrics]

    for d in domains:
        assert "precision" in d
        assert "recall" in d
        assert "false_positive_rate" in d
        assert "false_negative_rate" in d
        assert d["true_negatives"] == 1  # Correct negative classification on clean surface
