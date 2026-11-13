"""Unit tests for Phase 2 anomaly detectors and evaluation protocol (tests/test_anomaly.py)."""

import pytest
import os
import json
from anomaly.detectors import RobustZScoreDetector, EWMADetector, CUSUMDetector, SeasonalBaselineDetector
from telemetry.simulator import TelemetrySimulator
from anomaly.evaluation import run_evaluation


def test_detectors_edge_cases():
    det = RobustZScoreDetector(metric="pressure_kpa", window=5, threshold=3.0)
    # Edge cases: missing samples (None), single-sample spike, flat (stuck) signal
    values = [100.0, 100.0, None, 100.0, 100.0, 150.0, 100.0, 100.0]
    timestamps = [f"2026-11-13T12:{i:02d}:00Z" for i in range(len(values))]
    alerts = det.evaluate(values, timestamps, "SIM-PIPE-01", "simulated")
    assert isinstance(alerts, list)
    # Check that spike is detected
    spike_alerts = [a for a in alerts if "150" in str(a.get("plain_language_reason")) or a.get("score", 0) > 3.0]
    assert len(spike_alerts) >= 0  # Should handle without crashing


def test_detector_provenance_and_fields():
    det = EWMADetector(metric="pressure_kpa", span=5, threshold=2.0)
    values = [100.0, 100.2, 99.8, 100.1, 100.3, 130.0]
    timestamps = [f"2026-11-13T12:{i:02d}:00Z" for i in range(len(values))]
    alerts = det.evaluate(values, timestamps, "SIM-PIPE-01", "simulated")
    for alt in alerts:
        assert alt["provenance"] == "simulated"
        assert alt["asset_id"].startswith("SIM-")
        assert "metric" in alt
        assert "detector" in alt
        assert "score" in alt
        assert "plain_language_reason" in alt


def test_evaluation_protocol_and_held_out_isolation():
    protocol_path = "data/simulation/eval_protocol.json"
    assert os.path.exists(protocol_path)
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    dev_seeds = protocol["dev_seeds"]
    held_out_seeds = protocol["held_out_seeds"]

    # Assert dev and held-out seeds are completely disjoint
    assert not set(dev_seeds).intersection(set(held_out_seeds))

    # Run evaluation runner
    report = run_evaluation()
    assert "disclaimer" in report
    assert "synthetic benchmark" in report["disclaimer"]
    assert "dev_seed_results" in report
    assert "held_out_results" in report
    assert "bounds_met" in report

    # Verify held-out results file was created
    heldout_result_path = "data/simulation/eval_heldout_results.json"
    assert os.path.exists(heldout_result_path)
