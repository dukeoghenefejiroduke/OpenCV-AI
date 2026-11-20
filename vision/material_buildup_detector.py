"""Experimental Material Buildup / Dust & Clinker Deposit Detector (vision/material_buildup_detector.py).

Provides an optional experimental classical screening detector for industrial dust, clinker deposits,
and material accumulation on cement plant equipment surfaces. Off by default and clearly labelled unvalidated.
Results are kept strictly separate from corrosion metrics and do not trigger automated asset status flags.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_COATING_DAMAGE,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class MaterialBuildupDetector(BaseDetector):
    """Optional experimental screening detector for dust, clinker, and material buildup deposits."""

    detector_id: str = "material_buildup_detector"
    detector_name: str = "Experimental Material Buildup & Dust Deposit Screening Detector (Unvalidated)"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_COATING_DAMAGE
    supported_input_type: str = "BGR numpy array or image path"

    # Constants with engineering rationale
    CLAHE_CLIP_LIMIT: float = 2.0        # Contrast enhancement clip limit for textured powder/dust
    CLAHE_GRID_SIZE: tuple = (8, 8)      # Tile grid size for local illumination normalization
    MIN_AREA_RATIO: float = 0.0003       # Minimum area ratio (0.03% of image) to filter tiny dust specks
    MAX_AREA_RATIO: float = 0.30         # Maximum area ratio (30% of image) to reject giant background patches
    COLOR_LOWER_GRAY: int = 120          # Lower bound for dull grey/white clinker dust in HSV
    COLOR_UPPER_GRAY: int = 210          # Upper bound for brightness of dust deposits

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental material buildup screening on input image.

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
                "limitations": ["Input validation rejected image for material buildup screening."],
                "metadata": {}
            }

        rejection_audit = []
        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width

            # Illumination normalization via CLAHE
            lab = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_GRID_SIZE)
            l_norm = clahe.apply(l_channel)
            norm_lab = cv2.merge((l_norm, a_channel, b_channel))
            norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)

            gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Adaptive thresholding to isolate accumulation patches of dust/clinker
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 45, 8
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Restrict to foreground asset ROI if available
            try:
                from vision.inspection import compute_foreground_roi_mask
                roi_mask, _, _ = compute_foreground_roi_mask(decoded_img)
                closed = cv2.bitwise_and(closed, roi_mask)
            except Exception:
                pass

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            min_area = int(total_pixels * self.MIN_AREA_RATIO)
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
                position, pct_img = get_region_position_and_percentage([int(x), int(y), int(w), int(h)], (height, width), area)

                finding = {
                    "finding_id": f"build_{uuid.uuid4().hex[:8]}",
                    "region_id": f"BLD-{idx+1:03d}",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": self.defect_category,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "position": position,
                    "percentage_of_image": pct_img,
                    "contour_summary": f"Approx contour points count: {len(cnt)}",
                    "pixel_measurement": {
                        "buildup_area_pixels": int(area)
                    },
                    "evidence_description": f"material_buildup_candidate / dust_or_clinker_deposit {idx+1} ({position}): area {int(area)}px ({pct_img}% of image), bbox [{x}, {y}, {w}, {h}]. experimental unvalidated screening.",
                    "confidence": 0.3,  # experimental unvalidated
                    "review_status": "PENDING_REVIEW",
                    "limitations": [
                        "experimental_detector: material buildup detector is optional, off by default, and unvalidated on broad industrial datasets.",
                        "Dust deposits, powder coatings, clinker accumulation, and normal surface discoloration can be visually indistinguishable.",
                        "Requires manual site verification by cement plant operations team."
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
                    "Experimental unvalidated dust/clinker deposit screening detector.",
                    "Findings are provisional and require manual review."
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
