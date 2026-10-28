"""Crack Screening Detector for CementSight AI (vision/crack_detector.py).

Provides an independent, experimental classical computer-vision screening detector for linear
structural cracks, fractures, and joint lines using edge gradient analysis and geometric filtering.
Results are kept strictly separate from corrosion metrics and do not trigger automated asset status flags.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_CRACK,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED
)
from app.validation import validate_image_input


class CrackDetector(BaseDetector):
    """Independent experimental crack screening detector using classical edge and contour geometry."""

    detector_id: str = "crack_screening_detector"
    detector_name: str = "Classical Structural Crack Screening Detector"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_CRACK
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental crack screening on the input image.

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
                "limitations": ["Input validation rejected image for crack screening."],
                "metadata": {}
            }

        try:
            height, width = decoded_img.shape[:2]
            gray = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Conservative Canny edge detection for structural line features
            edges = cv2.Canny(blurred, 50, 150)

            # Morphological closing to connect adjacent edge fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            total_crack_pixels = 0

            for idx, cnt in enumerate(contours):
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                # Filter for elongated, significant linear or jagged edge structures (perimeter filter)
                if perimeter > 30 and area < 0.2 * (height * width):
                    x, y, w, h = cv2.boundingRect(cnt)
                    total_crack_pixels += int(area)

                    finding = {
                        "finding_id": f"crack_{uuid.uuid4().hex[:8]}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "pixel_measurement": {
                            "contour_area_pixels": int(area),
                            "contour_perimeter_pixels": round(perimeter, 2)
                        },
                        "evidence_description": f"Linear crack candidate {idx+1}: perimeter {perimeter:.1f}px, bbox [{x}, {y}, {w}, {h}].",
                        "confidence": None,  # Experimental screening detector does not calculate calibrated ML confidence
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "Experimental screening detector: linear features may include scratches, welds, seams, shadows, or background texture.",
                            "Pixel measurements are uncalibrated (no physical millimeters without camera calibration)."
                        ]
                    }
                    findings.append(finding)

            execution_status = STATUS_CANDIDATE_DETECTED if len(findings) > 0 else STATUS_NO_CANDIDATE
            total_pixels = height * width
            crack_coverage_pct = (total_crack_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

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
                    "Experimental screening detector; high false positive rate expected on textured surfaces, weld seams, and scratched paint.",
                    "Does not affect asset safety status or corrosion recommendations."
                ],
                "metadata": {
                    "total_crack_pixels": total_crack_pixels,
                    "estimated_crack_coverage_percentage": round(crack_coverage_pct, 3),
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
                "limitations": ["Crack screening detector failed due to an unexpected runtime exception."],
                "metadata": {}
            }
