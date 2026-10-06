"""Deterministic Agentic Decision Layer for CementSight AI.

Consumes structured inspection results from the vision pipeline (vision/inspection.py)
and applies deterministic rules to determine operational decisions (CRITICAL, WARNING, PASS),
proposed actions, approval requirements, and detailed reasoning without LLMs, randomness,
or external network calls.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np

from vision.inspection import inspect_corrosion

# Clear constants for decision thresholds matching vision/inspection.py
CRITICAL_THRESHOLD: float = 10.0
WARNING_THRESHOLD: float = 2.0


@dataclass
class InspectionDecision:
    """Structured decision output produced by the CementSight agent layer."""
    decision: str  # "CRITICAL" | "WARNING" | "PASS"
    action: str
    approval_required: bool
    reason: str
    inspection_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision dataclass to dictionary format."""
        return asdict(self)


def evaluate_inspection(report: Optional[Dict[str, Any]]) -> InspectionDecision:
    """Evaluates a vision inspection report deterministically and produces an agent decision.

    Consumes inspection output containing corrosion_percentage, region_count, recommendation, etc.
    Safely handles malformed or missing data by falling back to safe defaults.
    """
    if not isinstance(report, dict):
        return InspectionDecision(
            decision="WARNING",
            action="Proposed scheduling a routine maintenance inspection due to malformed or missing inspection data.",
            approval_required=False,
            reason="Inspection report was missing or malformed; defaulting to conservative warning state.",
            inspection_summary={"raw_report": report}
        )

    # Extract metrics defensively with fallback defaults
    corrosion_percentage = float(report.get("corrosion_percentage", 0.0))
    region_count = int(report.get("region_count", 0))
    recommendation = str(report.get("recommendation", "")).upper()

    # Determine decision state based on rules and thresholds
    if recommendation == "CRITICAL" or corrosion_percentage >= CRITICAL_THRESHOLD:
        decision = "CRITICAL"
        action = "Proposed immediate emergency maintenance work-order request. Awaiting supervisor approval."
        approval_required = True
        reason = (
            f"Critical condition detected: corrosion percentage ({corrosion_percentage}%) "
            f"meets or exceeds critical threshold ({CRITICAL_THRESHOLD}%) with {region_count} region(s) identified."
        )
    elif recommendation == "WARNING" or corrosion_percentage >= WARNING_THRESHOLD:
        decision = "WARNING"
        action = "Proposed scheduling a routine maintenance inspection and ongoing monitoring."
        approval_required = False
        reason = (
            f"Warning-level condition detected: corrosion percentage ({corrosion_percentage}%) "
            f"is within the warning range ({WARNING_THRESHOLD}% to {CRITICAL_THRESHOLD}%) with {region_count} region(s) identified."
        )
    else:
        decision = "PASS"
        action = "Recorded normal equipment status. No maintenance escalation required."
        approval_required = False
        reason = (
            f"Normal condition: corrosion percentage ({corrosion_percentage}%) "
            f"is below the warning threshold ({WARNING_THRESHOLD}%) with {region_count} region(s) identified."
        )

    # Build concise inspection summary subset
    inspection_summary = {
        "corrosion_percentage": corrosion_percentage,
        "region_count": region_count,
        "recommendation": recommendation if recommendation else ("CRITICAL" if corrosion_percentage >= CRITICAL_THRESHOLD else ("WARNING" if corrosion_percentage >= WARNING_THRESHOLD else "PASS")),
        "dimensions": report.get("dimensions", {"width": 0, "height": 0}),
        "bounding_boxes": report.get("bounding_boxes", [])
    }

    return InspectionDecision(
        decision=decision,
        action=action,
        approval_required=approval_required,
        reason=reason,
        inspection_summary=inspection_summary
    )


def inspect_and_decide(
    image_path_or_array: Union[str, np.ndarray],
    output_annotated_path: Optional[str] = None,
    output_roi_path: Optional[str] = None,
    use_roi: bool = True
) -> InspectionDecision:
    """Integration helper: runs OpenCV vision inspection and immediately evaluates agent decision."""
    report = inspect_corrosion(
        image_path_or_array,
        output_annotated_path=output_annotated_path,
        output_roi_path=output_roi_path,
        use_roi=use_roi
    )
    return evaluate_inspection(report)
