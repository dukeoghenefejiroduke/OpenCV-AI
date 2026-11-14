"""Unit tests for Phase 4 Multi-Hazard Risk Fusion (tests/test_risk_fusion.py)."""

import pytest
from risk.engine import RiskFusionEngine


def test_risk_critical_visual():
    visual = {"corrosion_percentage": 20.0, "recommendation": "CRITICAL", "confirmed": True}
    res = RiskFusionEngine.evaluate_risk(visual_report=visual, anomalies=[], provenance="real")
    assert res["risk_level"] == "CRITICAL"
    assert res["approval_required"] is True
    assert res["dispatched"] is False
    assert len(res["contributions"]) > 0


def test_risk_high_visual():
    visual = {"corrosion_percentage": 10.0, "recommendation": "WARNING", "confirmed": True}
    res = RiskFusionEngine.evaluate_risk(visual_report=visual, anomalies=[], provenance="real")
    assert res["risk_level"] == "HIGH"
    assert res["approval_required"] is True
    assert res["dispatched"] is False


def test_risk_simulated_anomaly():
    anomalies = [{"detector": "ewma", "provenance": "simulated"}]
    res = RiskFusionEngine.evaluate_risk(visual_report={}, anomalies=anomalies, provenance="simulated")
    assert res["risk_level"] == "MEDIUM"
    assert res["dispatched"] is False


def test_risk_imported_real_cap():
    anomalies = [
        {"detector": "zscore", "provenance": "imported_real"},
        {"detector": "cusum", "provenance": "imported_real"}
    ]
    visual = {"corrosion_percentage": 25.0, "recommendation": "CRITICAL", "confirmed": False}
    res = RiskFusionEngine.evaluate_risk(visual_report=visual, anomalies=anomalies, provenance="imported_real", human_approval=False)
    # Without confirmed visual or human approval, cannot reach CRITICAL
    assert res["risk_level"] != "CRITICAL"
    assert res["attention_flag"] is True
    assert res["dispatched"] is False


def test_risk_low_baseline():
    res = RiskFusionEngine.evaluate_risk(visual_report={}, anomalies=[], provenance="real")
    assert res["risk_level"] == "LOW"
    assert res["dispatched"] is False
    assert res["approval_required"] is False
