"""Adaptive Multi-Space Rust Screening Detector for CementSight AI (vision/rust_detector.py).

Implements an advanced adaptive rust screening detector using gray-world white balance,
CLAHE enhancement in Lab space, combined Lab/HSV adaptive color likelihood, close-before-open
morphology, and relative area filtering. Registered alongside the frozen corrosion detector.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_CORROSION,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class RustDetector(BaseDetector):
    """Adaptive multi-space rust screening detector running alongside frozen corrosion inspection."""

    detector_id: str = "rust_detector"
    detector_name: str = "Adaptive Multi-Space Rust Screening Detector"
    version: str = "1.0.0"
    defect_category: str = CATEGORY_CORROSION
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs adaptive rust detection using Lab/HSV chrominance and adaptive thresholding.

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
                "limitations": ["Input validation rejected image."],
                "metadata": {}
            }

        try:
            h, w = decoded_img.shape[:2]
            total_pixels = h * w

            # 1. Gray-World White Balance
            b_ch, g_ch, r_ch = cv2.split(decoded_img)
            b_mean = float(np.mean(b_ch))
            g_mean = float(np.mean(g_ch))
            r_mean = float(np.mean(r_ch))
            avg_mean = (b_mean + g_mean + r_mean) / 3.0

            if b_mean > 0 and g_mean > 0 and r_mean > 0:
                b_gain = avg_mean / b_mean
                g_gain = avg_mean / g_mean
                r_gain = avg_mean / r_mean
                balanced_b = cv2.multiply(b_ch, b_gain)
                balanced_g = cv2.multiply(g_ch, g_gain)
                balanced_r = cv2.multiply(r_ch, r_gain)
                balanced = cv2.merge([balanced_b, balanced_g, balanced_r])
                balanced = np.clip(balanced, 0, 255).astype(np.uint8)
            else:
                balanced = decoded_img.copy()

            # 2. Lab conversion + CLAHE on L channel (handles dim images)
            lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab)
            l_ch, a_ch, b_lab = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_clahe, a_ch, b_lab])
            enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

            # 3. Multi-space Rust Likelihood (Lab a*, b* and HSV)
            # Rust typically exhibits elevated a* (red-green) and b* (yellow-blue) chromaticity
            a_float = a_ch.astype(np.float32)
            b_float = b_lab.astype(np.float32)
            
            # Adaptive thresholds based on image statistics
            a_mean, a_std = np.mean(a_float), np.std(a_float)
            b_mean_val, b_std = np.mean(b_float), np.std(b_float)

            # Chromaticity condition: a* and b* significantly above mean
            lab_rust_mask = (a_float > (a_mean + 0.5 * max(a_std, 2.0))) & (b_float > (b_mean_val + 0.3 * max(b_std, 2.0)))

            # HSV condition for rust/warm hues
            hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            # Hue for rust is typically 5 to 30; adapt value threshold for dim images
            v_min = 20 if np.mean(v_ch) < 60 else 40
            hsv_rust_mask = (h_ch >= 3) & (h_ch <= 32) & (s_ch >= 35) & (v_ch >= v_min)

            combined_mask = (lab_rust_mask & hsv_rust_mask).astype(np.uint8) * 255

            # 4. Morphology: Close before Open to bridge gaps and clean noise
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

            # 5. Connected components / Contours with relative area filtering
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area_pixels = max(15, int(0.0002 * total_pixels))
            max_area_pixels = int(0.6 * total_pixels)

            findings = []
            rust_pixel_count = 0
            bounding_boxes = []

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if min_area_pixels <= area <= max_area_pixels:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    bounding_boxes.append([int(x), int(y), int(cw), int(ch)])
                    rust_pixel_count += int(area)

            rust_percentage = round((rust_pixel_count / float(total_pixels)) * 100.0, 3)

            for idx, bbox in enumerate(bounding_boxes):
                x, y, cw, ch = bbox
                area = cw * ch
                position, pct_img = get_region_position_and_percentage([x, y, cw, ch], (h, w), area)
                pct = round((area / float(total_pixels)) * 100.0, 3)

                finding = {
                    "finding_id": f"rust_cand_{uuid.uuid4().hex[:8]}",
                    "region_id": f"RST-{idx+1:03d}",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": self.defect_category,
                    "bbox": [x, y, cw, ch],
                    "position": position,
                    "percentage_of_image": pct_img,
                    "pixel_measurement": {
                        "bounding_box_area_pixels": int(area),
                        "estimated_percentage": pct,
                        "total_rust_percentage": rust_percentage
                    },
                    "evidence_description": f"Adaptive rust candidate {idx+1} ({position}): bbox [{x}, {y}, {cw}, {ch}], area {area}px ({pct_img}% of image). Screening candidate.",
                    "confidence": 0.45,
                    "review_status": "PENDING_REVIEW",
                    "limitations": [
                        "Adaptive multi-space rust screening candidate; manual verification required.",
                        "Lighting variations and surface staining can influence color-space likelihood."
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
                    "Adaptive Lab/HSV color-space screening detector.",
                    "Findings represent candidates requiring manual review."
                ],
                "metadata": {
                    "rust_percentage": rust_percentage,
                    "region_count": len(findings)
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
