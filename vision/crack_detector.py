"""Crack Screening Detector for CementSight AI (vision/crack_detector.py).

Provides an independent, experimental classical computer-vision screening detector for linear
structural cracks, fractures, and joint lines using CLAHE illumination normalization, blackhat
morphological filtering for thin dark lines, elongation/aspect-ratio checks, and straight-edge seam suppression.
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
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class CrackDetector(BaseDetector):
    """Independent experimental crack screening detector using CLAHE, blackhat filtering, and seam suppression."""

    detector_id: str = "crack_screening_detector"
    detector_name: str = "Classical Structural Crack Screening Detector"
    version: str = "2.0.0"
    defect_category: str = CATEGORY_CRACK
    supported_input_type: str = "BGR numpy array or image path"

    # Constants with engineering rationale
    CLAHE_CLIP_LIMIT: float = 2.0        # Contrast enhancement clip limit for local shadow/crack isolation
    CLAHE_GRID_SIZE: tuple = (8, 8)      # Tile grid size for local illumination normalization
    GAUSSIAN_BLUR_KERNEL: tuple = (5, 5) # Gaussian blur kernel to smooth high-frequency sensor noise
    CANNY_LOW_THRESH: int = 50           # Canny low gradient threshold for edge detection
    CANNy_HIGH_THRESH: int = 150         # Canny high gradient threshold for edge detection
    BLACKHAT_KERNEL_SIZE: tuple = (11, 11) # Structuring element size for blackhat filtering of dark linear features
    MIN_PERIMETER: float = 25.0          # Minimum perimeter to filter insignificant speck artifacts
    MAX_AREA_RATIO: float = 0.20         # Maximum area ratio (20% of image) to reject giant non-crack contours
    MIN_ASPECT_RATIO: float = 2.5        # Minimum aspect ratio (length/width or width/height) for elongated crack structures
    FRAME_ALIGNMENT_TOLERANCE: float = 0.90 # Fraction of image width/height indicating straight frame boundary/seam

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs experimental crack screening on the input image with CLAHE and seam suppression.

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

        rejection_audit = []
        try:
            height, width = decoded_img.shape[:2]
            total_pixels = height * width

            # Illumination normalization via CLAHE on L channel
            lab = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_GRID_SIZE)
            l_norm = clahe.apply(l_channel)
            norm_lab = cv2.merge((l_norm, a_channel, b_channel))
            norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)

            gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, self.GAUSSIAN_BLUR_KERNEL, 0)

            # Blackhat morphological filtering for thin dark lines / cracks against textured background
            bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.BLACKHAT_KERNEL_SIZE)
            blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, bh_kernel)

            # Edge detection combined with blackhat thresholding
            edges = cv2.Canny(blurred, self.CANNY_LOW_THRESH, self.CANNy_HIGH_THRESH)
            _, bh_thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_or(edges, bh_thresh)

            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_edges = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            findings = []
            max_area = int(total_pixels * self.MAX_AREA_RATIO)

            for idx, cnt in enumerate(contours):
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)

                if perimeter < self.MIN_PERIMETER:
                    rejection_audit.append({"reason": "below_min_perimeter", "perimeter": round(perimeter, 2), "min_required": self.MIN_PERIMETER})
                    continue
                if area > max_area:
                    rejection_audit.append({"reason": "exceeds_max_area", "area": area, "max_allowed": max_area})
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(max(w, h)) / float(min(w, h)) if min(w, h) > 0 else 1.0
                position, pct_img = get_region_position_and_percentage([int(x), int(y), int(w), int(h)], (height, width), area)

                # Check for straight edge aligned with frame (likely welding seam, formwork line, or boundary)
                is_straight_seam = (
                    (w >= self.FRAME_ALIGNMENT_TOLERANCE * width and h < 0.15 * height) or
                    (h >= self.FRAME_ALIGNMENT_TOLERANCE * height and w < 0.15 * width)
                )

                if is_straight_seam:
                    # Flag as possible seam / review rather than confirmed crack
                    finding = {
                        "finding_id": f"seam_{uuid.uuid4().hex[:8]}",
                        "region_id": f"SEAM-{idx+1:03d}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "position": position,
                        "percentage_of_image": pct_img,
                        "contour_summary": f"Approx contour points count: {len(cnt)}",
                        "pixel_measurement": {
                            "contour_area_pixels": int(area),
                            "contour_perimeter_pixels": round(perimeter, 2),
                            "aspect_ratio": round(aspect_ratio, 2)
                        },
                        "evidence_description": f"possible_seam_review {idx+1} ({position}): perimeter {perimeter:.1f}px, bbox [{x}, {y}, {w}, {h}]. Straight edge aligned with frame boundary; classified as potential weld seam or formwork line, not confirmed crack.",
                        "confidence": 0.2,  # low confidence unconfirmed seam
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "possible_seam_review: linear feature matches structural seam or formwork boundary rather than irregular fracture crack.",
                            "Requires manual verification."
                        ]
                    }
                    findings.append(finding)
                    continue

                # Filter for elongated crack structures
                if aspect_ratio >= self.MIN_ASPECT_RATIO:
                    finding = {
                        "finding_id": f"crack_{uuid.uuid4().hex[:8]}",
                        "region_id": f"CRK-{idx+1:03d}",
                        "detector_id": self.detector_id,
                        "detector_version": self.version,
                        "defect_category": self.defect_category,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "position": position,
                        "percentage_of_image": pct_img,
                        "contour_summary": f"Approx contour points count: {len(cnt)}",
                        "pixel_measurement": {
                            "contour_area_pixels": int(area),
                            "contour_perimeter_pixels": round(perimeter, 2),
                            "aspect_ratio": round(aspect_ratio, 2)
                        },
                        "evidence_description": f"Linear structural crack candidate {idx+1} ({position}): perimeter {perimeter:.1f}px, area {int(area)}px ({pct_img}% of image), aspect ratio {aspect_ratio:.2f}, bbox [{x}, {y}, {w}, {h}].",
                        "confidence": None,  # Experimental screening detector does not calculate calibrated ML confidence
                        "review_status": "PENDING_REVIEW",
                        "limitations": [
                            "experimental_screening_detector: classical edge gradient and blackhat screening detector.",
                            "Scratches, shadows, mortar joints, and surface textures can produce false-positive linear responses.",
                            "Pixel measurements are uncalibrated image-space estimates.",
                            "requires_manual_inspection: physical crack depth, width, and structural propagation cannot be determined from imagery alone."
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
                    "Experimental classical structural crack screening detector.",
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
                "limitations": ["Detector execution failed due to an unhandled runtime exception."],
                "metadata": {"rejection_audit": rejection_audit}
            }
