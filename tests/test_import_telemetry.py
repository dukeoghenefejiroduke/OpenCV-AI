"""Unit tests for Phase 6 Telemetry CSV Importer (tests/test_import_telemetry.py)."""

import pytest
import os
import csv
from tools.import_telemetry import import_telemetry_csv


def test_import_telemetry_success(tmp_path):
    csv_file = tmp_path / "sample_telemetry.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "timestamp", "metric", "value", "units", "scenario_id"])
        writer.writerow(["SEN-01", "2026-11-13T12:00:00Z", "pressure_kpa", "101.5", "kPa", "test_scen"])

    records = import_telemetry_csv(
        str(csv_file),
        source_url="https://example.com/dataset.csv",
        license_str="MIT",
        retrieved_date="2026-11-13"
    )
    assert len(records) == 1
    assert records[0]["provenance"] == "imported_real"
    assert records[0]["source_url"] == "https://example.com/dataset.csv"
    assert records[0]["license"] == "MIT"
    assert records[0]["retrieved_date"] == "2026-11-13"


def test_import_telemetry_refusal_missing_metadata(tmp_path):
    csv_file = tmp_path / "sample_telemetry.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "timestamp", "metric", "value", "units", "scenario_id"])
        writer.writerow(["SEN-01", "2026-11-13T12:00:00Z", "pressure_kpa", "101.5", "kPa", "test_scen"])

    # Missing source_url
    with pytest.raises(ValueError, match="source_url"):
        import_telemetry_csv(str(csv_file), source_url="", license_str="MIT", retrieved_date="2026-11-13")

    # Missing license
    with pytest.raises(ValueError, match="license"):
        import_telemetry_csv(str(csv_file), source_url="https://example.com", license_str="", retrieved_date="2026-11-13")

    # Missing retrieved_date
    with pytest.raises(ValueError, match="retrieved_date"):
        import_telemetry_csv(str(csv_file), source_url="https://example.com", license_str="MIT", retrieved_date="")
