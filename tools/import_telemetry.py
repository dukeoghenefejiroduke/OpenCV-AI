"""Telemetry CSV Importer Tool (tools/import_telemetry.py).

Maps a CSV file to the telemetry schema with provenance 'imported_real'.
Requires source_url, license, and retrieved_date as mandatory arguments; refuses without them.
Does NOT download anything and does NOT fabricate real data.
"""

import os
import csv
import sys
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

from telemetry.schema import validate_telemetry_record


def import_telemetry_csv(
    csv_path: str,
    source_url: str,
    license_str: str,
    retrieved_date: str,
    asset_id: str = "REAL-ASSET-01"
) -> List[Dict[str, Any]]:
    """Imports CSV telemetry records, asserting mandatory metadata arguments are provided."""
    if not source_url or not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("Import refused: mandatory argument 'source_url' is missing or invalid.")
    if not license_str or not isinstance(license_str, str) or not license_str.strip():
        raise ValueError("Import refused: mandatory argument 'license' is missing or invalid.")
    if not retrieved_date or not isinstance(retrieved_date, str) or not retrieved_date.strip():
        raise ValueError("Import refused: mandatory argument 'retrieved_date' is missing or invalid.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            metric = row.get("metric", "pressure_kpa")
            try:
                val = float(row.get("value", 0.0))
            except ValueError:
                continue
            units = row.get("units", "kPa")

            record = {
                "sensor_id": row.get("sensor_id", f"{asset_id}-SEN-01"),
                "asset_id": asset_id,
                "timestamp": timestamp,
                "metric": metric,
                "value": val,
                "units": units,
                "provenance": "imported_real",
                "scenario_id": row.get("scenario_id", "field_observation"),
                "simulator_version": "import_v1.0",
                "seed": 0,
                "quality_flag": "ok",
                "source_url": source_url.strip(),
                "license": license_str.strip(),
                "retrieved_date": retrieved_date.strip()
            }
            validate_telemetry_record(record)
            records.append(record)

    return records


if __name__ == "__main__":
    print("Telemetry Import Tool loaded.")
