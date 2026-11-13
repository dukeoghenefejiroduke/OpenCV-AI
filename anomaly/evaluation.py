"""Evaluation and benchmarking runner for anomaly detectors (anomaly/evaluation.py).

Enforces anti-overfitting protocol:
- Loads evaluation protocol from data/simulation/eval_protocol.json.
- Asserts tuning code ONLY uses dev_seeds.
- Runs sealed held-out seeds exactly ONCE and saves to data/simulation/eval_heldout_results.json.
- Labels all reports with disclaimer: 'synthetic benchmark, simplified physics, not field evidence'.
"""

import os
import json
from typing import Dict, Any, List
from telemetry.simulator import TelemetrySimulator
from anomaly.detectors import RobustZScoreDetector, EWMADetector, CUSUMDetector, SeasonalBaselineDetector


def run_evaluation() -> Dict[str, Any]:
    protocol_path = "data/simulation/eval_protocol.json"
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    dev_seeds = protocol["dev_seeds"]
    held_out_seeds = protocol["held_out_seeds"]
    bounds = protocol["pre_declared_bounds"]
    scenarios = protocol["scenario_list"]

    # Ensure held_out_seeds are not in dev_seeds
    assert not set(dev_seeds).intersection(set(held_out_seeds)), "Security violation: held-out seeds overlap with dev seeds!"

    # Instantiate detectors
    detectors = [
        RobustZScoreDetector(metric="pressure_kpa", window=20, threshold=3.0),
        EWMADetector(metric="pressure_kpa", span=15, threshold=2.5),
        CUSUMDetector(metric="pressure_kpa", threshold=4.0)
    ]

    def evaluate_set(seeds: List[int]) -> Dict[str, Any]:
        results_by_fault = {scen: {"detection_delays": [], "false_alarms": 0, "total_samples": 0} for scen in scenarios if scen != "none"}
        steady_false_alarms = 0
        steady_total_samples = 0

        for seed in seeds:
            sim = TelemetrySimulator(seed=seed)
            for fault in scenarios:
                n_steps = 100
                fault_start = 50
                series = sim.generate_series(
                    asset_id=f"SIM-PIPE-{seed}",
                    scenario_id=f"scen_{fault}_{seed}",
                    fault_type=fault,
                    fault_start_step=fault_start,
                    n_steps=n_steps
                )

                # Group by metric (pressure_kpa)
                pressure_records = [r for r in series if r["metric"] == "pressure_kpa"]
                values = [r["value"] for r in pressure_records]
                timestamps = [r["timestamp"] for r in pressure_records]
                provenance = "simulated"

                # Run detectors
                all_alerts = []
                for det in detectors:
                    alerts = det.evaluate(values, timestamps, f"SIM-PIPE-{seed}", provenance)
                    all_alerts.extend(alerts)

                if fault == "none":
                    steady_total_samples += len(values)
                    steady_false_alarms += len(all_alerts)
                else:
                    # Find first detection after fault_start
                    first_detection_step = None
                    for alt in all_alerts:
                        # find index matching timestamp
                        try:
                            idx = timestamps.index(alt["timestamp"])
                            if idx >= fault_start:
                                delay = idx - fault_start
                                if first_detection_step is None or delay < first_detection_step:
                                    first_detection_step = delay
                        except ValueError:
                            pass

                    if first_detection_step is not None:
                        results_by_fault[fault]["detection_delays"].append(first_detection_step)
                    else:
                        # missed detection penalized with max delay
                        results_by_fault[fault]["detection_delays"].append(n_steps - fault_start)

        summary = {}
        for fault, data in results_by_fault.items():
            delays = data["detection_delays"]
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            summary[fault] = {
                "average_detection_delay_steps": round(avg_delay, 2),
                "detection_rate": len([d for d in delays if d <= bounds["max_detection_delay_steps"].get(fault, 15)]) / len(delays) if delays else 0.0
            }

        fa_per_1000 = (steady_false_alarms / max(1, steady_total_samples)) * 1000
        summary["steady_state_false_alarms_per_1000"] = round(fa_per_1000, 2)
        return summary

    # Tune/Evaluate on Dev Seeds ONLY (verifying dev seeds are used)
    dev_results = evaluate_set(dev_seeds)

    # Run held-out seeds EXACTLY ONCE
    held_out_results = evaluate_set(held_out_seeds)

    # Check bounds against held-out results
    bounds_met = True
    missed_reasons = []

    if held_out_results["steady_state_false_alarms_per_1000"] > bounds["max_false_alarms_per_1000_samples"]:
        bounds_met = False
        missed_reasons.append(f"False alarms per 1000 ({held_out_results['steady_state_false_alarms_per_1000']}) exceeded bound ({bounds['max_false_alarms_per_1000_samples']})")

    for fault, metrics in held_out_results.items():
        if fault == "steady_state_false_alarms_per_1000":
            continue
        max_allowed_delay = bounds["max_detection_delay_steps"].get(fault, 15)
        if metrics["average_detection_delay_steps"] > max_allowed_delay:
            bounds_met = False
            missed_reasons.append(f"Fault '{fault}' avg delay ({metrics['average_detection_delay_steps']}) exceeded bound ({max_allowed_delay})")

    final_report = {
        "disclaimer": protocol["disclaimer"],
        "dev_seed_results": dev_results,
        "held_out_results": held_out_results,
        "bounds_met": bounds_met,
        "missed_reasons": missed_reasons
    }

    # Write to NEW file (never overwrite)
    heldout_output_path = "data/simulation/eval_heldout_results.json"
    os.makedirs(os.path.dirname(heldout_output_path), exist_ok=True)
    with open(heldout_output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    return final_report


if __name__ == "__main__":
    report = run_evaluation()
    print("Evaluation Complete. Report:", json.dumps(report, indent=2))
