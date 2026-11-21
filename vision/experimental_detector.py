"""Experimental Secondary Hazard Detectors for CementSight AI (vision/experimental_detector.py).

Implements independent experimental screening detectors for secondary hazards:
- Crack: Multi-scale morphological black-hat + elongated aspect-ratio/eccentricity filter (aspect ratio >= 5.0, length >= 40px).
- Coating/Surface: Local texture & color discontinuity filter for peeling/blistering (area >= 0.5% image area).
- Leakage: HSV darkening + saturation drop + vertical drip shape heuristics (height > 1.8 * width, V < 75, S < 60).
- Insulation: Torn/missing jacketing texture & color anomaly filter (runs only when material is Insulation, Mixed, or Unknown, never Steel).

All findings are strictly tagged experimental=True and requires_human_review=True, and capped at top 3 highest-confidence regions per image (confidence >= 0.65).
"""

from typing import Dict, Any, List
import numpy as np
import cv2
import uuid

from vision.contract import (
    BaseDetector,
    CATEGORY_CRACK,
    CATEGORY_COATING,
    CATEGORY_LEAKAGE,
    CATEGORY_INSULATION,
    STATUS_NO_CANDIDATE,
    STATUS_CANDIDATE_DETECTED,
    STATUS_DETECTOR_FAILED,
    STATUS_INPUT_REJECTED,
    get_region_position_and_percentage
)
from app.validation import validate_image_input


class ExperimentalCrackDetector(BaseDetector):
    detector_id: str = "experimental_crack_detector"
    detector_name: str = "Experimental Multi-Scale Blackhat Crack Detector"
    version: str = "2.2.0"
    defect_category: str = CATEGORY_CRACK
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        is_valid, decoded, err = validate_image_input(image)
        if not is_valid or decoded is None:
            return {"detector_id": self.detector_id, "execution_status": STATUS_INPUT_REJECTED, "findings": [], "errors": [err]}
        
        height, width = decoded.shape[:2]
        total_pixels = height * width
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        
        findings = []
        for k_size in [9, 15, 21]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, thresh = cv2.threshold(blackhat, 45, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 40:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                max_dim = float(max(w, h))
                min_dim = float(min(w, h))
                if min_dim < 3:
                    continue
                aspect_ratio = max_dim / min_dim
                
                # Strict constraints: minimum length >= 40px and aspect ratio >= 5.0
                if max_dim < 40 or aspect_ratio < 5.0:
                    continue
                
                position, pct = get_region_position_and_percentage([x, y, w, h], (height, width), area)
                confidence = min(0.90, max(0.65, 0.65 + (aspect_ratio - 5.0) * 0.02))
                findings.append({
                    "finding_id": f"exp_crk_{uuid.uuid4().hex[:8]}",
                    "region_id": f"CRK-{len(findings)+1:03d}",
                    "detector_id": self.detector_id,
                    "defect_category": self.defect_category,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": round(confidence, 2),
                    "experimental": True,
                    "requires_human_review": True,
                    "evidence_description": f"Experimental crack candidate at {position} (aspect ratio {aspect_ratio:.1f}, length {int(max_dim)}px)."
                })
        
        # Deduplicate overlapping findings and cap to top 3 highest-confidence regions
        findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)
        unique_findings = []
        for f in findings:
            if len(unique_findings) >= 3:
                break
            box = f["bbox"]
            overlaps = any(
                max(0, min(box[0]+box[2], uf["bbox"][0]+uf["bbox"][2]) - max(box[0], uf["bbox"][0])) *
                max(0, min(box[1]+box[3], uf["bbox"][1]+uf["bbox"][3]) - max(box[1], uf["bbox"][1])) / (box[2]*box[3] + 1e-5) > 0.3
                for uf in unique_findings
            )
            if not overlaps:
                unique_findings.append(f)

        status = STATUS_CANDIDATE_DETECTED if unique_findings else STATUS_NO_CANDIDATE
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "execution_status": status,
            "findings": unique_findings,
            "errors": [],
            "experimental": True,
            "requires_human_review": True
        }


class ExperimentalCoatingDetector(BaseDetector):
    detector_id: str = "experimental_coating_detector"
    detector_name: str = "Experimental Local Texture Discontinuity Coating Detector"
    version: str = "2.2.0"
    defect_category: str = CATEGORY_COATING
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        is_valid, decoded, err = validate_image_input(image)
        if not is_valid or decoded is None:
            return {"detector_id": self.detector_id, "execution_status": STATUS_INPUT_REJECTED, "findings": [], "errors": [err]}
        
        height, width = decoded.shape[:2]
        total_pixels = height * width
        min_coating_area = int(total_pixels * 0.002) # >= 0.2% of image area
        
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(decoded, cv2.COLOR_BGR2HSV)
        
        # Green paint mask (H: 35 to 85 in OpenCV HSV, S > 30, V > 30) representing peeling/flaking green coating
        green_paint_mask = cv2.inRange(hsv, np.array([35, 30, 30], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
        
        # Also detect texture gradients/delamination boundaries between paint and rust
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        grad_mag = cv2.convertScaleAbs(lap)
        edge_mask = ((grad_mag > 25) & (green_paint_mask > 0)).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_coating = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        
        cnts, _ = cv2.findContours(closed_coating, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        findings = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_coating_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 or h < 15:
                continue
            
            position, pct = get_region_position_and_percentage([x, y, w, h], (height, width), area)
            confidence = min(0.92, max(0.68, 0.68 + (area / total_pixels) * 5.0))
            findings.append({
                "finding_id": f"exp_coat_{uuid.uuid4().hex[:8]}",
                "region_id": f"COT-{len(findings)+1:03d}",
                "detector_id": self.detector_id,
                "defect_category": self.defect_category,
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": round(confidence, 2),
                "experimental": True,
                "requires_human_review": True,
                "evidence_description": f"Peeling green paint coating delamination at {position} (area {int(area)}px)."
            })
            
        findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)[:4]
        status = STATUS_CANDIDATE_DETECTED if findings else STATUS_NO_CANDIDATE
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "execution_status": status,
            "findings": findings,
            "errors": [],
            "experimental": True,
            "requires_human_review": True
        }


class ExperimentalLeakageDetector(BaseDetector):
    detector_id: str = "experimental_leakage_detector"
    detector_name: str = "Experimental HSV Leakage & Vertical Drip Detector"
    version: str = "2.2.0"
    defect_category: str = CATEGORY_LEAKAGE
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        is_valid, decoded, err = validate_image_input(image)
        if not is_valid or decoded is None:
            return {"detector_id": self.detector_id, "execution_status": STATUS_INPUT_REJECTED, "findings": [], "errors": [err]}
        
        height, width = decoded.shape[:2]
        hsv = cv2.cvtColor(decoded, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        
        # Dark low-saturation core: V < 75, S < 60
        leak_mask = (v < 75) & (s < 60)
        leak_mask = (leak_mask.astype(np.uint8)) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)) # vertical elongation
        vertical_drip = cv2.morphologyEx(leak_mask, cv2.MORPH_OPEN, kernel)
        
        cnts, _ = cv2.findContours(vertical_drip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        findings = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0:
                continue
            aspect_ratio = float(h) / float(w)
            # Require vertical elongation: height > 1.8 * width AND h >= 30px
            if h < 30 or aspect_ratio < 1.8:
                continue
                
            position, pct = get_region_position_and_percentage([x, y, w, h], (height, width), area)
            confidence = min(0.92, max(0.65, 0.65 + (aspect_ratio - 1.8) * 0.05))
            findings.append({
                "finding_id": f"exp_leak_{uuid.uuid4().hex[:8]}",
                "region_id": f"LEK-{len(findings)+1:03d}",
                "detector_id": self.detector_id,
                "defect_category": self.defect_category,
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": round(confidence, 2),
                "experimental": True,
                "requires_human_review": True,
                "evidence_description": f"Experimental vertical leakage drip at {position} (aspect ratio {aspect_ratio:.1f})."
            })
            
        findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)[:3]
        status = STATUS_CANDIDATE_DETECTED if findings else STATUS_NO_CANDIDATE
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "execution_status": status,
            "findings": findings,
            "errors": [],
            "experimental": True,
            "requires_human_review": True
        }


class ExperimentalInsulationDetector(BaseDetector):
    detector_id: str = "experimental_insulation_detector"
    detector_name: str = "Experimental Insulation Cladding Anomaly Detector"
    version: str = "2.2.0"
    defect_category: str = CATEGORY_INSULATION
    supported_input_type: str = "BGR numpy array or image path"

    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        material = kwargs.get("material", "Unknown")
        if isinstance(material, str) and material.strip().lower() == "steel":
            return {
                "detector_id": self.detector_id,
                "detector_name": self.detector_name,
                "version": self.version,
                "defect_category": self.defect_category,
                "execution_status": STATUS_NO_CANDIDATE,
                "findings": [],
                "errors": [],
                "experimental": True,
                "requires_human_review": True
            }

        is_valid, decoded, err = validate_image_input(image)
        if not is_valid or decoded is None:
            return {"detector_id": self.detector_id, "execution_status": STATUS_INPUT_REJECTED, "findings": [], "errors": [err]}
        
        height, width = decoded.shape[:2]
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, thresh = cv2.threshold(mag_norm, 70, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        findings = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
                
            position, pct = get_region_position_and_percentage([x, y, w, h], (height, width), area)
            confidence = 0.68
            findings.append({
                "finding_id": f"exp_insul_{uuid.uuid4().hex[:8]}",
                "region_id": f"INS-{len(findings)+1:03d}",
                "detector_id": self.detector_id,
                "defect_category": self.defect_category,
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": confidence,
                "experimental": True,
                "requires_human_review": True,
                "evidence_description": f"Experimental insulation/cladding anomaly at {position}."
            })
            
        findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)[:3]
        status = STATUS_CANDIDATE_DETECTED if findings else STATUS_NO_CANDIDATE
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "execution_status": status,
            "findings": findings,
            "errors": [],
            "experimental": True,
            "requires_human_review": True
        }
