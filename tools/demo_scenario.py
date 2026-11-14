"""End-to-End Condition Alerting Demo Scenario Script (tools/demo_scenario.py).

Builds a SIM network graph, injects a leak scenario via TelemetrySimulator,
runs anomaly detectors, fuses risk via RiskFusionEngine, and exports a report with SIMULATED banners.
"""

import os
import json
from datetime import datetime, timezone
from telemetry.simulator import TelemetrySimulator
from anomaly.detectors import RobustZScoreDetector
from twin.graph import DigitalTwinGraph
from risk.engine import RiskFusionEngine


def run_demo_scenario(seed: int = 42) -> dict:
    print("=== DeltaTwin Condition Alerting Demo Scenario ===")
    
    # 1. Build SIM network graph
    graph_data = {
        "namespace": "SIM",
        "badge": "[SIMULATED]",
        "nodes": [
            {"id": "node_SIM_PIPE_01", "health_status": "CRITICAL"},
            {"id": "node_SIM_VALVE_01", "health_status": "WARNING"}
        ],
        "segments": [{"source": "node_SIM_PIPE_01", "target": "node_SIM_VALVE_01"}],
        "assets": [{"id": "SIM-PIPE-01", "health_status": "CRITICAL"}]
    }
    twin = DigitalTwinGraph(graph_data)
    health_agg = twin.aggregate_and_propagate_health()
    print(f"Network Health Aggregation: {json.dumps(health_agg, indent=2)}")

    # 2. Inject leak telemetry
    sim = TelemetrySimulator(seed=seed, simulator_version="v1.1")
    series = sim.generate_series(asset_id="SIM-PIPE-01", fault_type="leak", fault_start_step=40, n_steps=80)
    print(f"Generated {len(series)} simulated telemetry records with leak fault.")

    # 3. Run detectors on pressure metric
    pressure_vals = [r["value"] for r in series if r["metric"] == "pressure_kpa"]
    timestamps = [r["timestamp"] for r in series if r["metric"] == "pressure_kpa"]
    detector = RobustZScoreDetector(metric="pressure_kpa", window=15, threshold=2.5)
    alerts = detector.evaluate(pressure_vals, timestamps, "SIM-PIPE-01", "simulated")
    print(f"Detected {len(alerts)} anomalies.")

    # 4. Fuse risk
    visual_report = {"corrosion_percentage": 18.5, "recommendation": "CRITICAL", "confirmed": True}
    risk_res = RiskFusionEngine.evaluate_risk(visual_report=visual_report, anomalies=alerts, provenance="simulated")
    print(f"Fused Risk Assessment: {json.dumps(risk_res, indent=2)}")

    # 5. Export demo report
    demo_report = {
        "badge": "[SIMULATED] - Condition Alerting Demo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "network_health": health_agg,
        "anomalies_detected": len(alerts),
        "risk_assessment": risk_res,
        "disclaimer": "synthetic benchmark, simplified physics, not field evidence"
    }

    out_path = "data/outputs/demo_scenario_report.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo_report, f, indent=2)

    print(f"Demo scenario report exported to {out_path}")
    return demo_report


if __name__ == "__main__":
    run_demo_scenario()
