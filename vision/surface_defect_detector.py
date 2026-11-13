"""Surface-Defect & Coating Damage Screening Detector for CementSight AI (vision/surface_defect_detector.py).

Provides an independent experimental classical computer-vision screening detector for coating
blistering, peeling, flaking, and concrete spalling defects using local intensity variation,
texture roughness analysis, circularity filtering, and contour geometry.
Results are kept strictly separate from corrosion metrics and do not trigger automated asset status flags.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_COATING_DAMAGE,
    CATEGORY_SPALLING,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class SurfaceDefectDetector(BaseDetector):
    """Independent experimental screening detector for coating blistering, peeling, and concrete spalling."""

    detector_id: str = "surface_defect_detector"
    detector_name: str = "Classical Surface Defect Screening Detector (Coating Damage & Spalling)"
    version: str = "2.0.0"
    defect_category: str = CATEGORY_COATING_DAMAGE
    supported_input_type: str = "BGR numpy array or image path"

    # Constants with engineering rationale
    CLAHE_CLIP_LIMIT: float = 2.0        # CLAHE clip limit for enhancing surface texture contrast
    CLAHE_GRID_SIZE: tuple = (8, 8)      # CLAHE grid tile size
    MIN_BLOB_AREA_RATIO: float = 0.0001  # Minimum area ratio (0.01% of image) to filter micro-specks
    MAX_BLOB_AREA_RATIO: float = 0.15    # Maximum area ratio (15% of image) to reject giant background regions
    CIRCULARITY_BLISTER_THRESH: float = 0.55 # Circularity threshold distinguishing rounded blisters/peeling from irregular spalls

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental coating blistering, peeling, and concrete spalling screening on input image.

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
                "limitations": ["Input validation rejected image for surface defect screening."],
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

            # Local contrast enhancement / adaptive thresholding to detect local surface anomalies
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 5
            )

            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            min_blob_area = max(50, int(total_pixels * self.MIN_BLOB_AREA_RATIO))
            max_blob_area = int(total_pixels * self.MAX_BLOB_AREA_RATIO)

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_blob_area:
                    rejection_audit.append({"reason": "below_min_blob_area", "area": area, "min_required": min_blob_area})
                    continue
                if area > max_blob_area:
                    rejection_audit.append({"reason": "exceeds_max_blob_area", "area": area, "max_allowed": max_blob_area})
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 1.0
                position, pct_img = get_region_position_and_percentage([int(x), int(y), int(w), int(h)], (height, width), area)

                # Differentiate coating blistering/peeling vs concrete spalling
                if circularity > self.CIRCULARITY_BLISTER_THRESH and 0.4 <= aspect_ratio <= 2.5:
                    defect_cat = CATEGORY_COATING_DAMAGE
                    desc_type = "Coating Blister / Peeling Candidate"
                else:
                    defect_cat = CATEGORY_SPALLING
                    desc_type = "Concrete Spalling / Flaking Candidate"

                finding = {
                    "finding_id": f"surf_{uuid.uuid4().hex[:8]}",
                    "region_id": f"SURF-{idx+1:03d}",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": defect_cat,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "position": position,
                    "percentage_of_image": pct_img,
                    "contour_summary": f"Approx contour points count: {len(cnt)}",
                    "pixel_measurement": {
                        "blob_area_pixels": int(area),
                        "circularity": round(circularity, 3),
                        "aspect_ratio": round(aspect_ratio, 3)
                    },
                    "evidence_description": f"{desc_type} {idx+1} ({position}): area {int(area)}px ({pct_img}% of image), circularity {circularity:.2f}, bbox [{x}, {y}, {w}, {h}].",
                    "confidence": None,
                    "review_status": "PENDING_REVIEW",
                    "limitations": [
                        f"{desc_type}: experimental classical local intensity anomaly screening detector.",
                        "Surface stains, paint discoloration, lighting reflections, and dust can mimic coating blisters or concrete spalling.",
                        "Pixel measurements are uncalibrated image-space estimates.",
                        "requires_manual_inspection: physical adhesion loss and concrete depth cannot be determined from visual imagery alone."
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
                    "Experimental classical surface defect screening detector.",
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
