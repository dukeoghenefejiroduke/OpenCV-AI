"""Corrosion Detector Wrapper for CementSight AI (vision/corrosion_detector.py).

Wraps the established vision/inspection.py corrosion inspection pipeline into the shared
BaseDetector contract without modifying vision/inspection.py.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_CORROSION,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED
)
from vision.inspection import inspect_corrosion
from app.validation import validate_image_input


class CorrosionDetector(BaseDetector):
    """Modular corrosion detector wrapper preserving exact production logic and sample metrics."""

    detector_id: str = "corrosion_detector"
    detector_name: str = "Classical Corrosion & Rust Inspector"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_CORROSION
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs corrosion inspection and translates results into standard detector contract format.

        Args:
            image: BGR numpy array or valid image path.
            **kwargs: Additional parameters (e.g., output_annotated_path, use_roi).

        Returns:
            Standard detector result dictionary.
        """
        # Validate input using validation layer
        is_valid, decoded_img, err_msg = validate_image_input(image)
        if not is_valid or decoded_img is None:
            return {
                "detector_id": self.detector_id,
                "detector_name": self.detector_name,
                "version": self.version,
                "defect_category": self.defect_category,
                "supported_input_type": self.supported_input_type,
                "execution_status": STATUS_INPUT_REJECTED,
                "findings": [],
                "errors": [err_msg],
                "limitations": [
                    "Input validation rejected image (below minimum 128x128 resolution or invalid format)."
                ],
                "metadata": {}
            }

        try:
            output_annotated_path = kwargs.get("output_annotated_path", None)
            use_roi = kwargs.get("use_roi", True)

            inspection_report = inspect_corrosion(
                decoded_img,
                output_annotated_path=output_annotated_path,
                use_roi=use_roi
            )

            bounding_boxes = inspection_report.get("bounding_boxes", [])
            corrosion_pct = inspection_report.get("corrosion_percentage", 0.0)
            recommendation = inspection_report.get("recommendation", "PASS")
            total_pixels = inspection_report.get("total_pixels", 1)

            findings = []
            for idx, bbox in enumerate(bounding_boxes):
                x, y, w, h = bbox
                area = w * h
                pct = (area / total_pixels) * 100.0 if total_pixels > 0 else 0.0

                finding = {
                    "finding_id": f"corr_{uuid.uuid4().hex[:8]}",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": self.defect_category,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "pixel_measurement": {
                        "bounding_box_area_pixels": int(area),
                        "estimated_percentage": round(pct, 3),
                        "total_corrosion_percentage": corrosion_pct
                    },
                    "evidence_description": f"Corrosion bounding box {idx+1}: [{x}, {y}, {w}, {h}], approx area {area} pixels.",
                    "confidence": None,  # Classical computer vision screening estimate, not a ML probability score
                    "review_status": "PENDING_REVIEW",
                    "limitations": [
                        "Corrosion percentage is a computer-vision screening estimate, not a certified physical measurement.",
                        "Classical structural ROI segmentation depends on visible equipment structure/edges."
                    ]
                }
                findings.append(finding)

            execution_status = STATUS_CANDIDATE_DETECTED if len(findings) > 0 and recommendation != "PASS" else STATUS_NO_CANDIDATE
            if recommendation == "PASS" and len(findings) == 0:
                execution_status = STATUS_NO_CANDIDATE

            return {
                "detector_id": self.detector_id,
                "detector_name": self.detector_name,
                "version": self.version,
                "defect_category": self.defect_category,
                "supported_input_type": self.supported_input_type,
                "execution_status": execution_status,
                "findings": findings,
                "errors": [],
                "limitations": [
                    "Classical HSV color thresholding depends on lighting conditions and rust hue variations.",
                    "Screening estimate; manual verification required for maintenance scheduling."
                ],
                "metadata": {
                    "inspection_report": inspection_report
                }
            }

        except Exception as e:
            return {
                "detector_id": self.detector_id,
                "detector_name": self.detector_name,
                "version": self.version,
                "defect_category": self.defect_category,
                "supported_input_type": self.supported_input_type,
                "execution_status": STATUS_DETECTOR_FAILED,
                "findings": [],
                "errors": [str(e)],
                "limitations": ["Detector execution failed due to an unexpected runtime exception."],
                "metadata": {}
            }
