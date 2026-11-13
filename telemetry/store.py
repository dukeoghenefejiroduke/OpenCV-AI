"""Append-only telemetry store with strict separation from the real workspace store (telemetry/store.py).

Stores simulation data exclusively in data/simulation/, enforcing a 5,000-record retention cap
and rejecting any simulated records from being written into the real workspace store.
"""

import os
import json
from typing import List, Dict, Any, Optional

from telemetry.schema import validate_telemetry_record

DEFAULT_SIMULATION_ROOT = "data/simulation"
MAX_SIMULATION_RECORDS = 5000


class SimulationStore:
    def __init__(self, root: str = DEFAULT_SIMULATION_ROOT, max_records: int = MAX_SIMULATION_RECORDS):
        self.root = root
        self.max_records = max_records
        os.makedirs(self.root, exist_ok=True)
        self.storage_file = os.path.join(self.root, "telemetry_sim.jsonl")

    def append_records(self, records: List[Dict[str, Any]]) -> None:
        """Appends validated simulation records with a 5,000-record retention cap."""
        validated = []
        for rec in records:
            validate_telemetry_record(rec)
            if not rec["asset_id"].startswith("SIM-"):
                raise ValueError(f"Simulation asset ID must start with 'SIM-', got '{rec['asset_id']}'.")
            if rec["provenance"] != "simulated":
                raise ValueError(f"Simulation store only accepts provenance 'simulated', got '{rec['provenance']}'.")
            validated.append(rec)

        existing = self.list_all_records()
        existing.extend(validated)

        # Enforce retention cap (rolling drop of oldest)
        if len(existing) > self.max_records:
            existing = existing[-self.max_records:]

        # Atomic write
        tmp_file = self.storage_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for rec in existing:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        os.replace(tmp_file, self.storage_file)

    def list_all_records(self) -> List[Dict[str, Any]]:
        """Lists all stored simulation telemetry records."""
        if not os.path.exists(self.storage_file):
            return []
        records = []
        with open(self.storage_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def get_records_for_asset(self, asset_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.list_all_records() if r.get("asset_id") == asset_id]

    def clear(self) -> None:
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)


def refuse_simulated_in_real_workspace(store_instance: Any, simulated_record: Dict[str, Any]) -> None:
    """Refuses to write simulated records into the real workspace store."""
    if simulated_record.get("provenance") == "simulated" or str(simulated_record.get("asset_id", "")).startswith("SIM-"):
        raise ValueError("Security violation: Refusing to write simulated telemetry or simulated asset into the real workspace store.")
