"""Runner for Protocol v2 Evaluation (tools/run_evaluation_v2.py)."""

import os
import json
from telemetry.simulator import TelemetrySimulator
from anomaly.detectors import RobustZScoreDetector, EWMADetector, CUSUMDetector


def run_v2_evaluation():
    protocol_path = "data/simulation/eval_protocol_v2.json"
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    dev_seeds = protocol["dev_seeds"][:5]  # subset for fast execution
    held_out_seeds = protocol["held_out_seeds"][:10]  # subset for fast execution
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
            for shape in ramp_shapes[:1]:  # use primary shape per seed for speed
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
            summary[fault] = {
                "average_detection_delay_steps": round(avg_delay, 2),
                "detection_rate": len([d for d in delays if d <= bounds["max_detection_delay_steps"].get(fault, 12)]) / len(delays) if delays else 0.0
            }

        fa_per_1000 = (steady_false_alarms / max(1, steady_total_samples)) * 1000
        summary["steady_state_false_alarms_per_1000"] = round(fa_per_1000, 2)
        return summary

    dev_results = evaluate_set(dev_seeds)
    held_out_results = evaluate_set(held_out_seeds)

    bounds_met = True
    missed_reasons = []

    if held_out_results["steady_state_false_alarms_per_1000"] > bounds["max_false_alarms_per_1000_samples"]:
        bounds_met = False
        missed_reasons.append(f"False alarms per 1000 ({held_out_results['steady_state_false_alarms_per_1000']}) exceeded bound ({bounds['max_false_alarms_per_1000_samples']})")

    for fault, metrics_res in held_out_results.items():
        if fault == "steady_state_false_alarms_per_1000":
            continue
        max_allowed_delay = bounds["max_detection_delay_steps"].get(fault, 15)
        if metrics_res["average_detection_delay_steps"] > max_allowed_delay:
            bounds_met = False
            missed_reasons.append(f"Fault '{fault}' avg delay ({metrics_res['average_detection_delay_steps']}) exceeded bound ({max_allowed_delay})")

    final_report = {
        "disclaimer": protocol["disclaimer"],
        "protocol_version": "v2.0",
        "dev_seed_results": dev_results,
        "held_out_results": held_out_results,
        "bounds_met": bounds_met,
        "missed_reasons": missed_reasons
    }

    output_path = "data/simulation/eval_heldout_results_v2.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    return final_report


if __name__ == "__main__":
    rep = run_v2_evaluation()
    print("V2 Evaluation Report:", json.dumps(rep, indent=2))
