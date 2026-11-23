"""Regression & integration tests for DeltaTwin unified predictive maintenance fusion (tests/test_deltatwin_unified.py)."""

import pytest

from anomaly.detectors import RobustZScoreDetector, EWMADetector, CUSUMDetector
from risk.engine import RiskFusionEngine
from telemetry.simulator import TelemetrySimulator
from twin.graph import DigitalTwinGraph


def test_simulation_firewall_blocks_simulated_escalation_on_real_asset():
    """Simulated anomalies must never escalate a real asset's predictive risk assessment."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 1.0, "recommendation": "PASS", "confirmed": False},
        anomalies=[
            {"metric": "pressure_kpa", "detector": "ewma", "provenance": "simulated"},
            {"metric": "flow_m3h", "detector": "cusum", "provenance": "simulated"},
            {"metric": "temperature_c", "detector": "robust_zscore", "provenance": "simulated"},
        ],
        corrosion_history=[(0, 0.5), (10, 0.8), (20, 1.0)],
        provenance="simulated",
        is_real_asset=True,
    )
    assert res["risk_level"] == "LOW"
    assert res["anomaly_count"] == 0
    assert res["dispatched"] is False
    assert any("SIMULATION_FIREWALL" in c for c in res["contributions"])


def test_mixed_provenance_strips_simulated_anomalies_on_real_asset():
    """Even if caller passes simulated entries inside real anomalies list, firewall filters them out."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 8.0, "recommendation": "WARNING", "confirmed": True},
        anomalies=[
            {"metric": "pressure_kpa", "detector": "ewma", "provenance": "simulated"},
            {"metric": "flow_m3h", "detector": "cusum", "provenance": "simulated"},
        ],
        corrosion_history=[(0, 7.8), (10, 7.9), (20, 8.0)],
        provenance="imported_real",
        is_real_asset=True,
    )
    assert res["anomaly_count"] == 0
    assert res["dispatched"] is False


def test_velocity_withheld_below_three_observations():
    """Degradation velocity projection must be withheld when fewer than 3 dated inspections exist."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 6.0, "recommendation": "PASS", "confirmed": True},
        anomalies=[],
        corrosion_history=[(0, 2.0), (5, 6.0)],
        provenance="real",
        is_real_asset=True,
        critical_threshold=10.0,
    )
    assert res["degradation_velocity_pct_per_day"] == 0.0
    assert res["velocity_escalated"] is False
    assert res["dispatched"] is False
    assert any("FORECAST_INSUFFICIENT_DATA" in c for c in res["contributions"])


def test_velocity_advisory_escalates_confirmed_rapid_corrosion():
    """Confirmed visual corrosion growing toward critical within 14 days escalates to HIGH (never dispatched)."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 8.0, "recommendation": "PASS", "confirmed": True},
        anomalies=[],
        corrosion_history=[(0, 2.0), (5, 5.0), (10, 8.0)],
        provenance="real",
        is_real_asset=True,
        critical_threshold=10.0,
    )
    assert res["degradation_velocity_pct_per_day"] > 0.4
    assert res["estimated_days_to_critical"] is not None
    assert res["estimated_days_to_critical"] <= 14.0
    assert res["velocity_escalated"] is True
    assert res["risk_level"] == "HIGH"
    assert res["approval_required"] is True
    assert res["dispatched"] is False
    assert any("VELOCITY_ADVISORY" in c for c in res["contributions"])


def test_imported_real_capped_without_visual_or_human_approval():
    """Imported real anomalies alone cannot reach CRITICAL without confirmed visual or human approval."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 1.0, "recommendation": "PASS", "confirmed": False},
        anomalies=[
            {"metric": "pressure_kpa", "detector": "robust_zscore", "provenance": "imported_real"},
            {"metric": "flow_m3h", "detector": "ewma", "provenance": "imported_real"},
            {"metric": "vibration_mm_s", "detector": "cusum", "provenance": "imported_real"},
        ],
        corrosion_history=[(0, 1.0)],
        provenance="imported_real",
        human_approval=False,
        is_real_asset=True,
    )
    assert res["risk_level"] != "CRITICAL"
    assert res["attention_flag"] is True
    assert res["dispatched"] is False


def test_multimodal_confirmed_visual_plus_real_anomalies_escalates():
    """Confirmed visual finding plus >= 2 real telemetry anomalies corroborates escalation to HIGH."""
    res = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 4.0, "recommendation": "PASS", "confirmed": True},
        anomalies=[
            {"metric": "pressure_kpa", "detector": "robust_zscore", "provenance": "imported_real"},
            {"metric": "flow_m3h", "detector": "ewma", "provenance": "imported_real"},
        ],
        corrosion_history=[(0, 4.0)],
        provenance="imported_real",
        is_real_asset=True,
    )
    assert res["risk_level"] == "HIGH"
    assert res["approval_required"] is True
    assert res["dispatched"] is False
    assert any("CORRELATED_MULTIMODAL" in c for c in res["contributions"])


def test_end_to_end_deltatwin_sim_to_fusion_pipeline():
    """Links TelemetrySimulator -> Anomaly Detectors -> DigitalTwinGraph -> RiskFusionEngine (sandboxed)."""
    sim = TelemetrySimulator(seed=42)
    series = sim.generate_series(
        asset_id="SIM-PIPE-01",
        scenario_id="scen_leak_e2e",
        fault_type="leak",
        fault_start_step=30,
        n_steps=60,
        ramp_shape="linear",
    )
    p_recs = [r for r in series if r["metric"] == "pressure_kpa"]
    vals = [r["value"] for r in p_recs]
    tss = [r["timestamp"] for r in p_recs]

    z_alerts = RobustZScoreDetector("pressure_kpa").evaluate(vals, tss, "SIM-PIPE-01", "simulated")
    e_alerts = EWMADetector("pressure_kpa").evaluate(vals, tss, "SIM-PIPE-01", "simulated")
    c_alerts = CUSUMDetector("pressure_kpa").evaluate(vals, tss, "SIM-PIPE-01", "simulated")
    all_alerts = z_alerts + e_alerts + c_alerts
    assert len(all_alerts) > 0

    twin = DigitalTwinGraph({
        "namespace": "SIM",
        "nodes": [
            {"id": "SIM-PIPE-01", "health_status": "CRITICAL" if len(all_alerts) >= 2 else "WARNING"},
            {"id": "SIM-VALVE-02", "health_status": "HEALTHY"},
        ],
        "segments": [{"source": "SIM-PIPE-01", "target": "SIM-VALVE-02"}],
        "assets": [{"id": "SIM-PIPE-01"}],
    })
    twin.enforce_namespace_isolation({"PIPE-SOUTH-01", "PIPE-NORTH-02"})
    prop = twin.aggregate_and_propagate_health()
    assert prop["propagated_health"]["SIM-VALVE-02"] == "CRITICAL"
    assert prop["badge"] == "[SIMULATED]"

    fusion = RiskFusionEngine.evaluate_predictive_maintenance(
        visual_report={"corrosion_percentage": 9.3, "recommendation": "WARNING", "confirmed": True},
        anomalies=all_alerts,
        corrosion_history=[(0, 5.0), (5, 7.2), (10, 9.3)],
        provenance="simulated",
        is_real_asset=False,
    )
    assert fusion["risk_level"] in ("HIGH", "CRITICAL")
    assert fusion["dispatched"] is False
