"""Seeded telemetry simulator for DeltaTwin condition alerting demo (telemetry/simulator.py)."""

import random
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from telemetry.schema import validate_telemetry_record


class TelemetrySimulator:
    def __init__(self, seed: int = 42, simulator_version: str = "v1.0"):
        self.seed = seed
        self.simulator_version = simulator_version
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_series(
        self,
        asset_id: str = "SIM-PIPE-01",
        scenario_id: str = "scen_baseline",
        fault_type: str = "none",  # none, leak, bearing_wear, blockage, overheating
        fault_start_step: int = 50,
        n_steps: int = 100,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Generates a seeded telemetry time series with optional fault injection."""
        if start_time is None:
            start_time = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

        records = []
        # Base nominal values
        base_pressure = 100.0  # kPa
        base_flow = 50.0       # m3/h
        base_temp = 45.0       # C
        base_vibration = 2.0   # mm/s

        for step in range(n_steps):
            step_time = start_time + timedelta(minutes=15 * step)
            t_iso = step_time.isoformat(timespec="seconds").replace("+00:00", "Z")

            is_fault = step >= fault_start_step and fault_type != "none"

            # Add noise and drift
            p_noise = self.np_rng.normal(0, 1.0)
            f_noise = self.np_rng.normal(0, 0.5)
            t_noise = self.np_rng.normal(0, 0.2)
            v_noise = self.np_rng.normal(0, 0.1)

            p_val = base_pressure + p_noise
            f_val = base_flow + f_noise
            t_val = base_temp + t_noise
            v_val = base_vibration + v_noise

            # Apply fault injections
            if is_fault:
                if fault_type == "leak":
                    p_val -= 15.0 * min(1.0, (step - fault_start_step + 1) / 10.0)
                    f_val -= 8.0 * min(1.0, (step - fault_start_step + 1) / 10.0)
                elif fault_type == "bearing_wear":
                    v_val += 4.5 * min(1.0, (step - fault_start_step + 1) / 10.0)
                elif fault_type == "blockage":
                    p_val += 20.0 * min(1.0, (step - fault_start_step + 1) / 10.0)
                    f_val -= 12.0 * min(1.0, (step - fault_start_step + 1) / 10.0)
                elif fault_type == "overheating":
                    t_val += 18.0 * min(1.0, (step - fault_start_step + 1) / 10.0)

            # Create records for 4 metrics
            step_records = [
                {
                    "sensor_id": f"{asset_id}-PT-01",
                    "asset_id": asset_id,
                    "timestamp": t_iso,
                    "metric": "pressure_kpa",
                    "value": round(float(p_val), 2),
                    "units": "kPa",
                    "provenance": "simulated",
                    "scenario_id": scenario_id,
                    "simulator_version": self.simulator_version,
                    "seed": self.seed,
                    "quality_flag": "ok",
                    "ground_truth_fault": is_fault
                },
                {
                    "sensor_id": f"{asset_id}-FT-01",
                    "asset_id": asset_id,
                    "timestamp": t_iso,
                    "metric": "flow_m3h",
                    "value": round(float(f_val), 2),
                    "units": "m3/h",
                    "provenance": "simulated",
                    "scenario_id": scenario_id,
                    "simulator_version": self.simulator_version,
                    "seed": self.seed,
                    "quality_flag": "ok",
                    "ground_truth_fault": is_fault
                },
                {
                    "sensor_id": f"{asset_id}-TT-01",
                    "asset_id": asset_id,
                    "timestamp": t_iso,
                    "metric": "temperature_c",
                    "value": round(float(t_val), 2),
                    "units": "C",
                    "provenance": "simulated",
                    "scenario_id": scenario_id,
                    "simulator_version": self.simulator_version,
                    "seed": self.seed,
                    "quality_flag": "ok",
                    "ground_truth_fault": is_fault and fault_type == "overheating"
                },
                {
                    "sensor_id": f"{asset_id}-VT-01",
                    "asset_id": asset_id,
                    "timestamp": t_iso,
                    "metric": "vibration_mm_s",
                    "value": round(float(v_val), 2),
                    "units": "mm/s",
                    "provenance": "simulated",
                    "scenario_id": scenario_id,
                    "simulator_version": self.simulator_version,
                    "seed": self.seed,
                    "quality_flag": "ok",
                    "ground_truth_fault": is_fault and fault_type == "bearing_wear"
                },
            ]

            for rec in step_records:
                validate_telemetry_record(rec)
                records.append(rec)

        return records
