"""Phase 1 telemetry tests (tests/test_phase1_telemetry.py)."""

import pytest
from telemetry.schema import validate_telemetry_record
from telemetry.simulator import TelemetrySimulator
from telemetry.store import SimulationStore, refuse_simulated_in_real_workspace


def test_schema_valid_and_rejections():
    valid_rec = {
        "sensor_id": "SIM-PIPE-01-PT-01",
        "asset_id": "SIM-PIPE-01",
        "timestamp": "2026-11-13T12:00:00Z",
        "metric": "pressure_kpa",
        "value": 100.5,
        "units": "kPa",
        "provenance": "simulated",
        "scenario_id": "scen_01",
        "simulator_version": "v1.0",
        "seed": 42,
        "quality_flag": "ok",
        "ground_truth_fault": False
    }
    assert validate_telemetry_record(valid_rec) == valid_rec

    # Rejection cases
    invalid_provenance = {**valid_rec, "provenance": "cloud_ai"}
    with pytest.raises(ValueError, match="provenance"):
        validate_telemetry_record(invalid_provenance)

    missing_timestamp = {**valid_rec}
    del missing_timestamp["timestamp"]
    with pytest.raises(ValueError, match="timestamp"):
        validate_telemetry_record(missing_timestamp)

    sim_missing_gt = {**valid_rec}
    del sim_missing_gt["ground_truth_fault"]
    with pytest.raises(ValueError, match="ground_truth_fault"):
        validate_telemetry_record(sim_missing_gt)


def test_simulator_determinism():
    sim1 = TelemetrySimulator(seed=123)
    series1 = sim1.generate_series(asset_id="SIM-PIPE-01", scenario_id="s1", fault_type="leak", n_steps=20)

    sim2 = TelemetrySimulator(seed=123)
    series2 = sim2.generate_series(asset_id="SIM-PIPE-01", scenario_id="s1", fault_type="leak", n_steps=20)

    assert series1 == series2


def test_simulator_fault_types():
    sim = TelemetrySimulator(seed=42)
    for fault in ["none", "leak", "bearing_wear", "blockage", "overheating"]:
        series = sim.generate_series(asset_id="SIM-PIPE-01", scenario_id=f"scen_{fault}", fault_type=fault, fault_start_step=10, n_steps=30)
        assert len(series) > 0


def test_simulation_store_and_retention(tmp_path):
    store_root = tmp_path / "simulation_store"
    sim_store = SimulationStore(root=str(store_root), max_records=50)

    sim = TelemetrySimulator(seed=42)
    records = sim.generate_series(asset_id="SIM-PIPE-01", scenario_id="scen_ret", n_steps=20) # 20 steps * 4 metrics = 80 records

    sim_store.append_records(records)
    all_recs = sim_store.list_all_records()
    # Should be capped at max_records = 50
    assert len(all_recs) == 50


def test_isolation_from_real_workspace(tmp_path):
    sim_rec = {
        "sensor_id": "SIM-PIPE-01-PT-01",
        "asset_id": "SIM-PIPE-01",
        "timestamp": "2026-11-13T12:00:00Z",
        "metric": "pressure_kpa",
        "value": 100.5,
        "units": "kPa",
        "provenance": "simulated",
        "scenario_id": "scen_iso",
        "simulator_version": "v1.0",
        "seed": 42,
        "quality_flag": "ok",
        "ground_truth_fault": False
    }

    # Refuse simulated record in real workspace store
    with pytest.raises(ValueError, match="Security violation"):
        refuse_simulated_in_real_workspace(None, sim_rec)
