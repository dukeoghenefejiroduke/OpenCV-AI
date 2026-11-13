"""Telemetry schema and validation for DeltaTwin condition alerting demo (telemetry/schema.py)."""

from datetime import datetime
from typing import Dict, Any, Optional

VALID_PROVENANCE = {"simulated", "imported_real"}
VALID_METRICS = {"pressure_kpa", "flow_m3h", "temperature_c", "vibration_mm_s"}


def validate_telemetry_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a telemetry record against the mandatory DeltaTwin schema.

    Raises ValueError with a clear error message on validation failure.
    """
    if not isinstance(record, dict):
        raise ValueError("Telemetry record must be a dictionary.")

    # Mandatory fields check
    sensor_id = record.get("sensor_id")
    if not sensor_id or not isinstance(sensor_id, str):
        raise ValueError("Field 'sensor_id' is mandatory and must be a non-empty string.")

    asset_id = record.get("asset_id")
    if not asset_id or not isinstance(asset_id, str):
        raise ValueError("Field 'asset_id' is mandatory and must be a non-empty string.")

    timestamp = record.get("timestamp")
    if not timestamp or not isinstance(timestamp, str):
        raise ValueError("Field 'timestamp' is mandatory and must be a UTC ISO string.")
    try:
        # Verify UTC ISO format
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception as e:
        raise ValueError(f"Field 'timestamp' must be a valid ISO datetime string: {e}")

    metric = record.get("metric")
    if not metric or not isinstance(metric, str):
        raise ValueError("Field 'metric' is mandatory and must be a string.")

    value = record.get("value")
    if value is None or not isinstance(value, (int, float)):
        raise ValueError("Field 'value' is mandatory and must be a numeric value.")

    units = record.get("units")
    if not units or not isinstance(units, str):
        raise ValueError("Field 'units' is mandatory and must be a string.")

    provenance = record.get("provenance")
    if provenance not in VALID_PROVENANCE:
        raise ValueError(f"Field 'provenance' is mandatory and must be one of {VALID_PROVENANCE}, got '{provenance}'.")

    scenario_id = record.get("scenario_id")
    if not scenario_id or not isinstance(scenario_id, str):
        raise ValueError("Field 'scenario_id' is mandatory and must be a string.")

    simulator_version = record.get("simulator_version")
    if not simulator_version or not isinstance(simulator_version, str):
        raise ValueError("Field 'simulator_version' is mandatory and must be a string.")

    seed = record.get("seed")
    if seed is None or not isinstance(seed, int):
        raise ValueError("Field 'seed' is mandatory and must be an integer.")

    quality_flag = record.get("quality_flag", "ok")
    if not isinstance(quality_flag, str):
        raise ValueError("Field 'quality_flag' must be a string.")

    # Ground truth label required if simulated
    if provenance == "simulated":
        if "ground_truth_fault" not in record:
            raise ValueError("Field 'ground_truth_fault' is mandatory for simulated records.")
        if not isinstance(record["ground_truth_fault"], bool):
            raise ValueError("Field 'ground_truth_fault' must be a boolean.")

    return record
