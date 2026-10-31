"""Insulation Damage and Protective Cladding Screening Detector for CementSight AI (vision/insulation_detector.py).

Provides an independent experimental classical computer-vision screening detector for visible
insulation damage, missing/displaced external cladding, open seams, and gaps in insulation jackets.
Results are kept strictly separate from corrosion metrics and do not trigger automated asset status flags.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_INSULATION_DAMAGE,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED
)
from app.validation import validate_image_input


class InsulationDetector(BaseDetector):
    """Independent experimental screening detector for insulation damage and cladding anomalies."""

    detector_id: str = "insulation_detector"
    detector_name: str = "Classical Insulation Damage & Cladding Screening Detector"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_INSULATION_DAMAGE
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental insulation damage and cladding anomaly screening on input image.

        Args:
            image: BGR numpy array or valid image path.
            **kwargs: Additional parameters.

        Returns:
            Standard detector result dictionary.
        """
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
                "limitations": ["Input validation rejected image for insulation screening."],
                "metadata": {}
            }

        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width
            gray = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection combined with adaptive thresholding to isolate open seams, gaps, or cladding edges
            edges = cv2.Canny(blurred, 60, 180)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 35, 6
            )

            combined = cv2.bitwise_or(edges, thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            min_area = max(80, int(total_pixels * 0.00015))
            max_area = int(total_pixels * 0.20)

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 1.0

                x, y, w, h = cv2.boundingRect(cnt)

                # Cladding gaps / open seams often exhibit lower solidity or irregular bounding structures
                if solidity < 0.85 or (w > 0.05 * width and h > 0.05 * height):
                    finding = {
                        "finding_id": f"insul_{uuid.uuid4().hex[:8]}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "pixel_measurement": {
                            "region_area_pixels": int(area),
                            "solidity": round(solidity, 3)
                        },
                        "evidence_description": f"insulation_cladding_anomaly / open_seam_or_missing_cladding {idx+1}: area {int(area)}px, solidity {solidity:.2f}, bbox [{x}, {y}, {w}, {h}]. requires_manual_inspection.",
                        "confidence": None,  # Experimental screening detector does not calculate calibrated ML confidence
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "insulation_cladding_anomaly: experimental screening detector based on edge and contrast heuristics.",
                            "Ordinary RGB photographs do not establish thermal conductivity, insulation efficiency, heat loss, or operating temperature.",
                            "Visible gaps, color differences, or seams may represent normal construction joints, brackets, or reflections rather than physical insulation damage.",
                            "requires_manual_inspection: physical condition cannot be confirmed from image geometry alone."
                        ]
                    }
                    findings.append(finding)

            execution_status = STATUS_CANDIDATE_DETECTED if len(findings) > 0 else STATUS_NO_CANDIDATE

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
                    "Experimental screening detector; high false positive rate expected on mechanical joints, brackets, support straps, and paint variations.",
                    "RGB imaging cannot determine thermal performance or internal insulation degradation."
                ],
                "metadata": {
                    "total_insulation_candidates": len(findings),
                    "is_experimental": True
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
                "limitations": ["Insulation screening detector failed due to an unexpected runtime exception."],
                "metadata": {}
            }
