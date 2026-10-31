"""Surface-Defect Screening Detector for CementSight AI (vision/surface_defect_detector.py).

Provides an independent experimental classical computer-vision screening detector for coating
blistering and concrete spalling defects using local intensity variation, blob analysis,
circularity filtering, and contour geometry.
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
    STATUS_INPUT_REJECTED
)
from app.validation import validate_image_input


class SurfaceDefectDetector(BaseDetector):
    """Independent experimental screening detector for coating blistering and concrete spalling."""

    detector_id: str = "surface_defect_detector"
    detector_name: str = "Classical Surface Defect Screening Detector (Blistering & Spalling)"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_COATING_DAMAGE  # primary default category; findings can specify spalling or coating_damage
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental coating blistering and concrete spalling screening on input image.

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

        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width
            gray = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Local contrast enhancement / adaptive thresholding to detect local blobs (blisters / spalls)
            # Blisters tend to be rounded contrasting spots; spalls tend to be irregular chipped regions.
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 5
            )

            # Morphological cleaning to isolate localized blobs
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            min_blob_area = max(50, int(total_pixels * 0.0001))
            max_blob_area = int(total_pixels * 0.15)  # reject giant background regions

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_blob_area or area > max_blob_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # Calculate circularity: 4 * pi * area / (perimeter^2)
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 1.0

                # Differentiate potential coating blisters (more circular / convex) vs spalling (more irregular / angular)
                if circularity > 0.60 and 0.5 <= aspect_ratio <= 2.0:
                    defect_cat = CATEGORY_COATING_DAMAGE
                    desc_type = "Coating Blister Candidate"
                else:
                    defect_cat = CATEGORY_SPALLING
                    desc_type = "Concrete Spalling Candidate"

                finding = {
                    "finding_id": f"surf_{uuid.uuid4().hex[:8]}",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": defect_cat,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "pixel_measurement": {
                        "blob_area_pixels": int(area),
                        "circularity": round(circularity, 3),
                        "aspect_ratio": round(aspect_ratio, 3)
                    },
                    "evidence_description": f"{desc_type} {idx+1}: area {int(area)}px, circularity {circularity:.2f}, bbox [{x}, {y}, {w}, {h}].",
                    "confidence": None,  # Experimental screening detector does not calculate calibrated ML confidence
                    "review_status": "PENDING_REVIEW",
                    "limitations": [
                        "Experimental surface defect screening detector based on local intensity and blob geometry heuristics.",
                        "Circular blob geometry does not confirm a chemical coating blister; irregular shape does not confirm concrete spalling.",
                        "Pixel measurements are uncalibrated (no physical millimeters without camera calibration)."
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
                    "Experimental screening detector; high false positive rate expected on surface dirt, reflections, shadows, and texture variations.",
                    "Does not affect asset safety status or corrosion recommendations."
                ],
                "metadata": {
                    "total_surface_candidates": len(findings),
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
                "limitations": ["Surface defect detector failed due to an unexpected runtime exception."],
                "metadata": {}
            }
