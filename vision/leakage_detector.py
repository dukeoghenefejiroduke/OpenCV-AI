"""Leakage and Seepage Indicator Screening Detector for CementSight AI (vision/leakage_detector.py).

Provides an independent experimental classical computer-vision screening detector for potential
leakage indicators, vertical/diagonal seepage staining streaks, and irregular discoloration.
Results are kept strictly separate from corrosion metrics and do not trigger automated asset status flags.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_LEAKAGE,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED
)
from app.validation import validate_image_input


class LeakageDetector(BaseDetector):
    """Independent experimental screening detector for leakage and seepage indicators."""

    detector_id: str = "leakage_detector"
    detector_name: str = "Classical Leakage & Seepage Indicator Screening Detector"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_LEAKAGE
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental leakage and seepage indicator screening on input image.

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
                "limitations": ["Input validation rejected image for leakage screening."],
                "metadata": {}
            }

        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width
            gray = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Adaptive thresholding to detect darker regions, stains, or fluid streaks against lighter background
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 41, 7
            )

            # Morphological vertical / streak structuring to connect downward drainage / seepage paths
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
            streaked = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(streaked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            min_area = max(100, int(total_pixels * 0.0002))
            max_area = int(total_pixels * 0.25)

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / w if w > 0 else 1.0

                # Filter for vertically elongated streak structures or significant staining patches
                if aspect_ratio >= 1.3 or area > (total_pixels * 0.002):
                    finding = {
                        "finding_id": f"leak_{uuid.uuid4().hex[:8]}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "pixel_measurement": {
                            "streak_area_pixels": int(area),
                            "aspect_ratio": round(aspect_ratio, 3)
                        },
                        "evidence_description": f"potential_leakage_indicator / possible_seepage_staining {idx+1}: area {int(area)}px, aspect ratio {aspect_ratio:.2f}, bbox [{x}, {y}, {w}, {h}]. requires_manual_inspection.",
                        "confidence": None,  # Experimental screening detector does not calculate calibrated ML confidence
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "potential_leakage_indicator: experimental screening detector based on vertical gradient and staining heuristics.",
                            "possible_seepage_staining: dark vertical streaks may be shadows, oil marks, shadows, or background paint variations rather than active fluid leakage.",
                            "requires_manual_inspection: physical fluid composition, active leak rate, and structural confirmation cannot be established from a photograph alone.",
                            "Pixel measurements are uncalibrated."
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
                    "Experimental screening detector; high false positive rate expected on vertical shadows, pipe joints, and surface grime.",
                    "potential_leakage_indicator and possible_seepage_staining require manual inspection."
                ],
                "metadata": {
                    "total_leakage_candidates": len(findings),
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
                "limitations": ["Leakage screening detector failed due to an unexpected runtime exception."],
                "metadata": {}
            }
