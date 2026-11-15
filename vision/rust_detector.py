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

            # 1. Gray-World White Balance (skip if near neutral, or estimate from non-rust pixels)
            b_ch, g_ch, r_ch = cv2.split(decoded_img)
            b_mean = float(np.mean(b_ch))
            g_mean = float(np.mean(g_ch))
            r_mean = float(np.mean(r_ch))

            is_near_neutral = max(abs(b_mean - g_mean), abs(g_mean - r_mean), abs(b_mean - r_mean)) < 8.0
            if is_near_neutral:
                balanced = decoded_img.copy()
            else:
                lab_coarse = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2Lab)
                _, a_c, b_c = cv2.split(lab_coarse)
                non_rust_mask = (a_c < 135) | (b_c < 135)
                if np.sum(non_rust_mask) > 0.1 * total_pixels:
                    b_bg = b_ch[non_rust_mask]
                    g_bg = g_ch[non_rust_mask]
                    r_bg = r_ch[non_rust_mask]
                    b_mean_bg = float(np.mean(b_bg))
                    g_mean_bg = float(np.mean(g_bg))
                    r_mean_bg = float(np.mean(r_bg))
                    avg_mean = (b_mean_bg + g_mean_bg + r_mean_bg) / 3.0
                else:
                    avg_mean = (b_mean + g_mean + r_mean) / 3.0
                    b_mean_bg, g_mean_bg, r_mean_bg = b_mean, g_mean, r_mean

                if b_mean_bg > 0 and g_mean_bg > 0 and r_mean_bg > 0:
                    b_gain = avg_mean / b_mean_bg
                    g_gain = avg_mean / g_mean_bg
                    r_gain = avg_mean / r_mean_bg
                    balanced_b = cv2.multiply(b_ch, b_gain)
                    balanced_g = cv2.multiply(g_ch, g_gain)
                    balanced_r = cv2.multiply(r_ch, r_gain)
                    balanced = cv2.merge([balanced_b, balanced_g, balanced_r])
                    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
                else:
                    balanced = decoded_img.copy()

            # 2. Lab conversion + CLAHE on L channel
            lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab)
            l_ch, a_ch, b_lab = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_clahe, a_ch, b_lab])
            enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

            # 3. Absolute minimum criteria + adaptive tightening
            a_float = a_ch.astype(np.float32)
            b_float = b_lab.astype(np.float32)
            
            # Absolute criteria (Lab chroma distance from neutral & a*, b* elevation)
            chroma_dist = np.sqrt((a_float - 128.0)**2 + (b_float - 128.0)**2)
            absolute_lab_mask = (chroma_dist > 14.0) & (a_float > 131.0) & (b_float > 131.0)

            # Absolute HSV criteria
            hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            absolute_hsv_mask = (h_ch >= 3) & (h_ch <= 32) & (s_ch >= 40) & (v_ch >= 30)

            absolute_mask = absolute_lab_mask & absolute_hsv_mask

            # Adaptive tightening (never creates candidates, only restricts)
            a_mean, a_std = np.mean(a_float), np.std(a_float)
            b_mean_val, b_std = np.mean(b_float), np.std(b_float)
            adaptive_lab_mask = (a_float > (a_mean + 0.4 * max(a_std, 2.0))) & (b_float > (b_mean_val + 0.2 * max(b_std, 2.0)))

            v_min = 25 if np.mean(v_ch) < 60 else 45
            adaptive_hsv_mask = (h_ch >= 3) & (h_ch <= 32) & (s_ch >= 35) & (v_ch >= v_min)

            combined_mask = (absolute_mask & adaptive_lab_mask & adaptive_hsv_mask).astype(np.uint8) * 255

            # 4. Morphology: Close before Open
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

            # 5. Connected components / Contours with relative area filtering, roughness & shape stats
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area_pixels = max(15, int(0.0002 * total_pixels))
            max_area_pixels = int(0.6 * total_pixels)

            gray_full = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
            findings = []
            rust_pixel_count = 0
            bounding_boxes = []
            valid_contours = []

            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if min_area_pixels <= area <= max_area_pixels:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    bounding_boxes.append([int(x), int(y), int(cw), int(ch)])
                    valid_contours.append(cnt)
                    rust_pixel_count += int(area)

            rust_percentage = round((rust_pixel_count / float(total_pixels)) * 100.0, 3)

            for idx, bbox in enumerate(bounding_boxes):
                x, y, cw, ch = bbox
                cnt = valid_contours[idx] if idx < len(valid_contours) else None
                area = cw * ch

                perimeter = cv2.arcLength(cnt, True) if cnt is not None and len(cnt) > 0 else 1.0
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
                aspect_ratio = float(max(cw, ch)) / float(min(cw, ch)) if min(cw, ch) > 0 else 1.0
                hull = cv2.convexHull(cnt) if cnt is not None and len(cnt) > 0 else cnt
                hull_area = cv2.contourArea(hull) if hull is not None else area
                solidity = float(area) / float(hull_area) if hull_area > 0 else 1.0

                roi_gray = gray_full[max(0, y):min(h, y+ch), max(0, x):min(w, x+cw)]
                roughness = float(np.std(roi_gray)) if roi_gray.size > 0 else 0.0

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
                        "total_rust_percentage": rust_percentage,
                        "roughness_texture_std": round(roughness, 2),
                        "shape_stats": {
                            "circularity": round(circularity, 3),
                            "aspect_ratio": round(aspect_ratio, 3),
                            "solidity": round(solidity, 3)
                        }
                    },
                    "evidence_description": f"Adaptive rust candidate {idx+1} ({position}): bbox [{x}, {y}, {cw}, {ch}], area {area}px ({pct_img}% of image), roughness std {roughness:.1f}, circularity {circularity:.2f}. Screening candidate.",
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
                    "Adaptive Lab/HSV color-space screening detector with absolute chrominance and texture validation.",
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
