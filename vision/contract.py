"""Shared Detector Contract and Finding Schema for CementSight AI (vision/contract.py).

Defines the standardized protocol, execution statuses, defect categories, finding schemas,
and validation logic for all modular vision detectors (corrosion, cracks, spalling, etc.).
"""

from typing import Dict, Any, List, Optional, Tuple
import uuid
import numpy as np

# Execution Statuses
STATUS_NO_CANDIDATE = "NO_CANDIDATE"
STATUS_CANDIDATE_DETECTED = "CANDIDATE_DETECTED"
STATUS_DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"
STATUS_DETECTOR_FAILED = "DETECTOR_FAILED"
STATUS_INPUT_REJECTED = "INPUT_REJECTED"

# Defect Categories
CATEGORY_CORROSION = "corrosion"
CATEGORY_CRACK = "crack"
CATEGORY_SPALLING = "spalling"
CATEGORY_COATING_DAMAGE = "coating_damage"
CATEGORY_LEAKAGE = "leakage"
CATEGORY_INSULATION_DAMAGE = "insulation_damage"


def validate_finding(finding: Dict[str, Any], image_shape: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
    """Validates a single defect finding against the standard schema and coordinate constraints.

    Args:
        finding: Dictionary representing a defect finding.
        image_shape: Optional (height, width) tuple of the source image to validate bounding boxes.

    Returns:
        (is_valid, error_message)
    """
    required_keys = [
        "finding_id", "detector_id", "detector_version", "defect_category",
        "bbox", "pixel_measurement", "evidence_description", "confidence",
        "review_status", "limitations"
    ]
    for key in required_keys:
        if key not in finding:
            return False, f"Missing required finding key: '{key}'"

    # Validate bbox if present
    bbox = finding["bbox"]
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, "Invalid bbox format. Expected (x, y, w, h) list or tuple."
        x, y, w, h = bbox
        if not all(isinstance(v, (int, np.integer)) for v in [x, y, w, h]):
            return False, "Bounding box coordinates must be integers."
        if x < 0 or y < 0 or w < 0 or h < 0:
            return False, "Bounding box coordinates and dimensions must be non-negative."
        if not all(np.isfinite([x, y, w, h])):
            return False, "Bounding box values must be finite numbers."
        if image_shape is not None:
            img_h, img_w = image_shape
            if x + w > img_w + 2 or y + h > img_h + 2:  # allow 2px tolerance for rounding
                return False, f"Bounding box (x={x}, y={y}, w={w}, h={h}) exceeds image dimensions ({img_w}x{img_h})."

    # Validate pixel measurements
    meas = finding["pixel_measurement"]
    if meas is not None:
        if not isinstance(meas, dict):
            return False, "pixel_measurement must be a dictionary."
        for k, v in meas.items():
            if isinstance(v, (int, float, np.number)):
                if not np.isfinite(v):
                    return False, f"Non-finite value in pixel_measurement['{k}']: {v}"

    # Validate confidence
    conf = finding["confidence"]
    if conf is not None:
        if not isinstance(conf, (int, float, np.number)) or not np.isfinite(conf):
            return False, "Confidence must be a finite numeric value."
        if conf < 0.0 or conf > 1.0:
            return False, f"Confidence score ({conf}) must be between 0.0 and 1.0."

    return True, ""


class BaseDetector:
    """Base abstract class / protocol for modular hazard detectors."""

    detector_id: str = "base_detector"
    detector_name: str = "Base Detector"
    version: str = "1.0.0"
    defect_category: str = "unknown"
    supported_input_type: str = "BGR numpy array"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs detection on the input image.

        Must return a dictionary matching the standard detector result structure:
        {
            "detector_id": str,
            "detector_name": str,
            "version": str,
            "defect_category": str,
            "supported_input_type": str,
            "execution_status": str,
            "findings": List[Dict[str, Any]],
            "errors": List[str],
            "limitations": List[str],
            "metadata": Dict[str, Any]
        }
        """
        raise NotImplementedError("Detectors must implement the detect() method.")
