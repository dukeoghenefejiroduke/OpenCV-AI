"""Leakage and Seepage Indicator Screening Detector for CementSight AI (vision/leakage_detector.py).

Provides an independent experimental classical computer-vision screening detector for potential
leakage indicators, vertical/diagonal seepage staining streaks, and irregular discoloration
using CLAHE illumination normalization and morphological vertical streak structuring.
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
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class LeakageDetector(BaseDetector):
    """Independent experimental screening detector for leakage and seepage indicators using CLAHE."""

    detector_id: str = "leakage_detector"
    detector_name: str = "Classical Leakage & Seepage Indicator Screening Detector"
    version: str = "2.0.0"
    defect_category: str = CATEGORY_LEAKAGE
    supported_input_type: str = "BGR numpy array or image path"

    # Constants with engineering rationale
    CLAHE_CLIP_LIMIT: float = 2.0        # CLAHE clip limit for enhancing staining contrast
    CLAHE_GRID_SIZE: tuple = (8, 8)      # CLAHE grid size
    MIN_AREA_RATIO: float = 0.0002       # Minimum area ratio (0.02% of image) to filter minor spots
    MAX_AREA_RATIO: float = 0.25         # Maximum area ratio (25% of image) to reject large background zones
    MIN_ASPECT_RATIO: float = 1.3        # Minimum aspect ratio (height/width) for downward vertical drainage streaks

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

        rejection_audit = []
        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width

            # Illumination normalization via CLAHE
            lab = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_GRID_SIZE)
            l_norm = clahe.apply(l_chan)
            norm_lab = cv2.merge((l_norm, a_chan, b_chan))
            norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)

            gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
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
            min_area = max(100, int(total_pixels * self.MIN_AREA_RATIO))
            max_area = int(total_pixels * self.MAX_AREA_RATIO)

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_area:
                    rejection_audit.append({"reason": "below_min_area", "area": area, "min_required": min_area})
                    continue
                if area > max_area:
                    rejection_audit.append({"reason": "exceeds_max_area", "area": area, "max_allowed": max_area})
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / w if w > 0 else 1.0
                position, pct_img = get_region_position_and_percentage([int(x), int(y), int(w), int(h)], (height, width), area)

                # Filter for vertically elongated streak structures or significant staining patches
                if aspect_ratio >= self.MIN_ASPECT_RATIO or area > (total_pixels * 0.002):
                    finding = {
                        "finding_id": f"leak_{uuid.uuid4().hex[:8]}",
                        "region_id": f"LEAK-{idx+1:03d}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "position": position,
                        "percentage_of_image": pct_img,
                        "contour_summary": f"Approx contour points count: {len(cnt)}",
                        "pixel_measurement": {
                            "streak_area_pixels": int(area),
                            "aspect_ratio": round(aspect_ratio, 3)
                        },
                        "evidence_description": f"potential_leakage_indicator / possible_seepage_staining {idx+1} ({position}): area {int(area)}px ({pct_img}% of image), aspect ratio {aspect_ratio:.2f}, bbox [{x}, {y}, {w}, {h}]. requires_manual_inspection.",
                        "confidence": None,
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "potential_leakage_indicator: experimental screening detector based on vertical gradient and staining heuristics.",
                            "possible_seepage_staining: dark vertical streaks may be shadows, oil marks, shadows, or background paint variations rather than active fluid leakage.",
                            "requires_manual_inspection: physical fluid composition, active leak rate, and structural confirmation cannot be established from a photograph alone.",
                            "Pixel measurements are uncalibrated."
                        ]
                    }
                    findings.append(finding)
                else:
                    rejection_audit.append({"reason": "aspect_ratio_too_low", "aspect_ratio": round(aspect_ratio, 2), "min_required": self.MIN_ASPECT_RATIO})

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
                    "Experimental classical leakage and seepage screening detector.",
                    "Findings represent candidates requiring manual review."
                ],
                "metadata": {
                    "rejections_count": len(rejection_audit),
                    "rejection_audit": rejection_audit
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
                "limitations": ["Detector execution failed due to an unhandled exception."],
                "metadata": {"rejection_audit": rejection_audit}
            }
