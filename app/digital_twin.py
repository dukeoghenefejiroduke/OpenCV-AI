"""Digital Twin & Asset Hierarchy Mapping Module (app/digital_twin.py).

Links industrial inspection results and corrosion telemetry to plant hierarchy zones
(e.g., Plant Zone -> Equipment -> Asset ID) for digital twin visualization.
"""

from typing import Dict, Any


class AssetNode:
    """Represents a node in the industrial plant digital twin hierarchy."""

    def __init__(self, zone: str, equipment_type: str, asset_id: str, criticality: str = "HIGH"):
        self.zone = zone
        self.equipment_type = equipment_type
        self.asset_id = asset_id
        self.criticality = criticality
        self.last_inspection_record: Dict[str, Any] = {}

    def update_inspection(self, inspection_report: Dict[str, Any], decision_state: str) -> None:
        """Updates the asset node with the latest inspection results."""
        self.last_inspection_record = {
            "corrosion_percentage": inspection_report.get("corrosion_percentage", 0.0),
            "region_count": inspection_report.get("region_count", 0),
            "recommendation": inspection_report.get("recommendation", "PASS"),
            "decision": decision_state
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the digital twin asset record."""
        return {
            "zone": self.zone,
            "equipment_type": self.equipment_type,
            "asset_id": self.asset_id,
            "criticality": self.criticality,
            "last_inspection": self.last_inspection_record
        }
