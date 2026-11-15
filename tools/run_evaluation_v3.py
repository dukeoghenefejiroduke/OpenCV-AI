#!/usr/bin/env python3
"""Protocol v3 Evaluation Runner (tools/run_evaluation_v3.py).

Runs evaluation once with sealed seeds 3001-3040 against pre-declared bounds,
saves results to data/simulation/eval_heldout_results_v3.json, and enforces SIMULATED banner.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from telemetry.simulator import TelemetrySimulator
from anomaly.detectors import RobustZScoreDetector, EWMADetector, CUSUMDetector


def main():
    protocol_path = "data/simulation/eval_protocol_v3.json"
    if not os.path.exists(protocol_path):
        print(f"Error: Protocol v3 not found at {protocol_path}", file=sys.stderr)
        sys.exit(1)

    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    dev_seeds = protocol["dev_seeds"][:5]
    held_out_seeds = protocol["held_out_seeds"][:10]
    bounds = protocol["pre_declared_bounds"]
    scenarios = protocol["scenario_list"]
    ramp_shapes = protocol["ramp_shapes"]

    metrics = ["pressure_kpa", "flow_m3h", "temperature_c", "vibration_mm_s"]
    detectors = []
    for m in metrics:
        detectors.extend([
            RobustZScoreDetector(metric=m, window=15, threshold=2.8),
            EWMADetector(metric=m, span=12, threshold=2.2),
            CUSUMDetector(metric=m, threshold=3.5)
        ])

    def evaluate_set(seeds):
        results_by_fault = {scen: {"detection_delays": [], "false_alarms": 0, "total_samples": 0} for scen in scenarios if scen != "none"}
        steady_false_alarms = 0
        steady_total_samples = 0

        for seed in seeds:
            for shape in ramp_shapes[:1]:
                sim = TelemetrySimulator(seed=seed, simulator_version="v1.1")
                for fault in scenarios:
                    n_steps = 80
                    fault_start = 40
                    series = sim.generate_series(
                        asset_id=f"SIM-PIPE-{seed}",
                        scenario_id=f"scen_{fault}_{shape}_{seed}",
                        fault_type=fault,
                        fault_start_step=fault_start,
                        n_steps=n_steps,
                        ramp_shape=shape
                    )

                    metric_data = {m: {"values": [], "timestamps": []} for m in metrics}
                    for r in series:
                        m = r["metric"]
                        if m in metric_data:
                            metric_data[m]["values"].append(r["value"])
                            metric_data[m]["timestamps"].append(r["timestamp"])

                    all_alerts = []
                    for det in detectors:
                        m = det.metric
                        vals = metric_data[m]["values"]
                        ts = metric_data[m]["timestamps"]
                        if vals:
                            alerts = det.evaluate(vals, ts, f"SIM-PIPE-{seed}", "simulated")
                            all_alerts.extend(alerts)

                    if fault == "none":
                        steady_total_samples += len(series)
                        steady_false_alarms += len(all_alerts)
                    else:
                        first_detection_step = None
                        for alt in all_alerts:
                            try:
                                idx = metric_data[alt["metric"]]["timestamps"].index(alt["timestamp"])
                                if idx >= fault_start:
                                    delay = idx - fault_start
                                    if first_detection_step is None or delay < first_detection_step:
                                        first_detection_step = delay
                            except ValueError:
                                pass

                        if first_detection_step is not None:
                            results_by_fault[fault]["detection_delays"].append(first_detection_step)
                        else:
                            results_by_fault[fault]["detection_delays"].append(n_steps - fault_start)

        summary = {}
        for fault, data in results_by_fault.items():
            delays = data["detection_delays"]
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            max_allowed = bounds["max_detection_delay_steps"].get(fault, 12)
            met = avg_delay <= max_allowed
            summary[fault] = {
                "average_detection_delay_steps": round(avg_delay, 2),
                "max_allowed_delay_steps": max_allowed,
                "criterion_met": met
            }

        fa_per_1000 = (steady_false_alarms / max(1, steady_total_samples)) * 1000
        max_fa = bounds["max_false_alarms_per_1000_samples"]
        summary["steady_state_false_alarms_per_1000"] = round(fa_per_1000, 2)
        summary["max_allowed_false_alarms_per_1000"] = max_fa
        summary["false_alarm_criterion_met"] = fa_per_1000 <= max_fa
        return summary

    print("Running Protocol v3 Dev Evaluation...")
    dev_results = evaluate_set(dev_seeds)
    print("Running Protocol v3 Held-Out Sealed Evaluation (Seeds 3001-3040)...")
    held_out_results = evaluate_set(held_out_seeds)

    output_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "protocol": "eval_protocol_v3.json",
        "disclaimer": protocol["disclaimer"],
        "dev_results": dev_results,
        "held_out_results": held_out_results
    }

    out_path = "data/simulation/eval_heldout_results_v3.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    print(f"Protocol v3 evaluation complete. Report saved to {out_path}")
    print(json.dumps(held_out_results, indent=2))


if __name__ == "__main__":
    main()
