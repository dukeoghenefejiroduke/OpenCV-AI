"""Dataclasses for the inspection workspace."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


HUMAN_REVIEW_STATUSES = ("unreviewed", "confirmed", "rejected", "needs_followup", "pending_review", "retake_requested")


@dataclass
class Asset:
    asset_id: str            # user-supplied, unique
    location: str
    pipe_material: str = "unknown"
    pipe_type: str = "unknown"
    criticality: str = "normal"   # "low" | "normal" | "high" | "critical"
    notes: str = ""
    qr_payload: str = ""      # QR code or asset tag identifier payload


@dataclass
class Inspection:
    inspection_id: str
    asset_id: str
    inspection_date: str      # ISO 8601 date
    image_sha256: str         # SHA-256 of original image bytes
    image_ext: str            # ".jpg" | ".png" | ".webp"
    annotated_sha256: Optional[str] = None
    detector_results: Dict[str, Any] = field(default_factory=dict)
    authoritative_report: Dict[str, Any] = field(default_factory=dict)
    inspector_notes: str = ""
    created_at: str = ""


@dataclass
class Finding:
    finding_id: str
    inspection_id: str
    detector_id: str
    defect_category: str
    bbox: Optional[List[int]]
    confidence: Optional[float]     # heuristic_score; NOT a calibrated probability
    evidence_description: str
    limitations: List[str]
    human_review_status: str = "unreviewed"   # see HUMAN_REVIEW_STATUSES
    review_notes: str = ""
    reviewed_at: Optional[str] = None


@dataclass
class MaintenanceRecord:
    """Recorded maintenance work — distinct from recommendations."""
    record_id: str
    asset_id: str
    inspection_id: Optional[str]     # initial inspection showing defect
    work_performed: str
    performed_by: str
    performed_at: str
    kind: str = "recorded_work"      # "recorded_work" | "recommendation"
    verification_status: str = "pending" # "pending" | "verified_resolved" | "unresolved"
    followup_inspection_id: Optional[str] = None # follow-up inspection evidence after repair


@dataclass
class PriorityScore:
    category: str                # routine_review | follow_up | priority_maintenance | urgent_human_assessment
    score: int
    rationale: List[str]
    contributing_factors: Dict[str, Any]