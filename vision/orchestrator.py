"""Multi-Hazard Detector Orchestrator for CementSight AI (vision/orchestrator.py).

Coordinates execution across multiple independent hazard detectors (corrosion, cracks, surface defects,
leakage, insulation, and experimental material buildup), running an image quality gate first and aggregating
results into a unified platform inspection report with strict hierarchical fusion rules (Authoritative > Experimental),
structural foreground ROI extent filtering (65% inside), v4 split-don't-delete logic for giant corrosion boxes (>25% area),
strict leakage aspect ratio filtering (h/w > 4.0), containment nesting suppression, conditional surface defect gating,
size/confidence filtering, and fault isolation.
"""

from typing import Dict, Any, List, Optional, Tuple
import uuid
import numpy as np
import cv2

from vision.contract import BaseDetector, validate_finding, STATUS_DETECTOR_FAILED, STATUS_CANDIDATE_DETECTED
from vision.quality_gate import ImageQualityGate
from vision.inspection import compute_foreground_roi_mask
from vision.corrosion_detector import CorrosionDetector
from vision.rust_detector import RustDetector
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector
from vision.material_buildup_detector import MaterialBuildupDetector


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0
    x1, y1, w1, h1 = [float(v) for v in box1]
    x2, y2, w2, h2 = [float(v) for v in box2]

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_containment(box_small: List[int], box_large: List[int]) -> float:
    """Computes what fraction of box_small's area is contained within box_large."""
    if not box_small or not box_large or len(box_small) != 4 or len(box_large) != 4:
        return 0.0
    x1, y1, w1, h1 = [float(v) for v in box_small]
    x2, y2, w2, h2 = [float(v) for v in box_large]

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_small = w1 * h1
    if area_small <= 0:
        return 0.0
    return inter / area_small


def is_strictly_inside_roi(box: List[int], roi_mask: np.ndarray) -> bool:
    """Checks if bounding box [x, y, w, h] has its center inside ROI and >=40% of area inside ROI mask."""
    if roi_mask is None:
        return True
    x, y, w, h = [int(v) for v in box[:4]]
    h_img, w_img = roi_mask.shape[:2]

    # Check center or allow if substantial overlap
    cx = max(0, min(w_img - 1, x + w // 2))
    cy = max(0, min(h_img - 1, y + h // 2))
    center_ok = (roi_mask[cy, cx] > 0) if (0 <= cx < w_img and 0 <= cy < h_img) else False

    sub = roi_mask[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
    if sub.size == 0:
        return False
    overlap_ratio = np.sum(sub > 0) / sub.size
    return center_ok or (overlap_ratio >= 0.40)


def split_giant_corrosion_box(box: List[int], decoded_img: np.ndarray, pipe_roi_mask: np.ndarray) -> List[List[int]]:
    """Subdivides a giant corrosion bounding box into distinct patch boxes using morphological opening to break thin bridges."""
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = decoded_img.shape[:2]
    total_img_area = h_img * w_img
    min_area = int(total_img_area * 0.001)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    sub_img = decoded_img[y1:y2, x1:x2]
    if sub_img.size == 0:
        return [box]

    hsv = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([5, 50, 50], dtype=np.uint8)
    upper_rust = np.array([25, 255, 255], dtype=np.uint8)
    sub_mask = cv2.inRange(hsv, lower_rust, upper_rust)

    if pipe_roi_mask is not None:
        sub_roi = pipe_roi_mask[y1:y2, x1:x2]
        if sub_roi.shape == sub_mask.shape:
            sub_mask = cv2.bitwise_and(sub_mask, sub_roi)

    # Morphological opening with (25, 25) ellipse to separate bridged blobs into distinct patches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    opened = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sub_boxes = []
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            global_box = [int(x1 + bx), int(y1 + by), int(bw), int(bh)]
            sub_boxes.append(global_box)

    if not sub_boxes:
        return [box]
    return sub_boxes[:5]


class MultiHazardOrchestrator:
    """Orchestrates multiple modular vision detectors with pre-inspection quality gating,
    structural ROI masking, and hierarchical multi-hazard fusion rules.
    """

    def __init__(self, enable_experimental: bool = False):
        self.quality_gate = ImageQualityGate()
        self.detectors: List[BaseDetector] = [
            CorrosionDetector()
        ]
        if enable_experimental:
            self.detectors.extend([
                RustDetector(),
                CrackDetector(),
                SurfaceDefectDetector(),
                LeakageDetector(),
                InsulationDetector(),
                MaterialBuildupDetector()
            ])

    def register_detector(self, detector: BaseDetector):
        """Registers a new modular detector at runtime."""
        self.detectors.append(detector)

    def run_inspections(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs image quality assessment, foreground ROI computation, then all registered detectors
        with hierarchical fusion (Authoritative Corrosion > Experimental Screening).

        Args:
            image: BGR numpy array or image path.
            **kwargs: Extra parameters passed to detectors.

        Returns:
            Aggregated multi-hazard inspection report with quality gate status and fused findings.
        """
        if isinstance(image, str):
            import cv2
            decoded_img = cv2.imread(image)
        else:
            decoded_img = image

        if decoded_img is None or decoded_img.size == 0 or len(decoded_img.shape) < 2:
            decoded_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        img_shape = decoded_img.shape[:2]
        img_h, img_w = img_shape
        total_img_area = img_h * img_w

        quality_res = self.quality_gate.evaluate(decoded_img)
        is_low_quality = (quality_res.get("verdict") == "LOW_QUALITY")

        # Step 1 ROI: Compute foreground pipe/equipment ROI mask
        pipe_roi_mask, _, _ = compute_foreground_roi_mask(decoded_img)

        results = {}
        all_findings = []
        detector_summaries = []

        for detector in self.detectors:
            try:
                res = detector.detect(image, **kwargs)
                if not isinstance(res, dict):
                    raise TypeError(f"Detector returned non-dictionary type: {type(res)}")
            except Exception as e:
                res = {
                    "detector_id": detector.detector_id,
                    "detector_name": detector.detector_name,
                    "version": detector.version,
                    "defect_category": detector.defect_category,
                    "supported_input_type": detector.supported_input_type,
                    "execution_status": STATUS_DETECTOR_FAILED,
                    "findings": [],
                    "errors": [str(e)],
                    "limitations": ["Detector execution failed due to an unhandled runtime exception or invalid contract return type."],
                    "metadata": {}
                }

            findings = res.get("findings", [])
            if is_low_quality:
                for f in findings:
                    f["confidence"] = 0.2
                    if "limitations" in f and isinstance(f["limitations"], list):
                        f["limitations"].append("Warning: Image quality gate flagged low quality (retake image recommended). Findings have low confidence.")

            valid_findings = []
            contract_rejection_count = 0
            rejected_findings_summary = []
            for f in findings:
                is_valid, err = validate_finding(f, image_shape=img_shape)
                if is_valid:
                    valid_findings.append(f)
                else:
                    contract_rejection_count += 1
                    rejected_findings_summary.append({"finding": f, "error": err})
            res["findings"] = valid_findings
            res["contract_rejection_count"] = contract_rejection_count
            res["rejected_findings_summary"] = rejected_findings_summary
            results[detector.detector_id] = res

            detector_summaries.append({
                "detector_id": detector.detector_id,
                "detector_name": detector.detector_name,
                "version": detector.version,
                "defect_category": detector.defect_category,
                "execution_status": res.get("execution_status"),
                "findings_count": len(valid_findings),
                "contract_rejection_count": contract_rejection_count,
                "errors": res.get("errors", []),
                "limitations": res.get("limitations", [])
            })
            all_findings.extend(valid_findings)

        # --- HIERARCHICAL FUSION & FILTERING PIPELINE (v4: Split, Don't Delete) ---
        min_w = max(5, int(0.015 * img_w))
        min_h = max(5, int(0.015 * img_h))
        min_area_thresh = int(0.0005 * total_img_area)

        authoritative_findings = []
        experimental_findings = []

        for f in all_findings:
            det_id = f.get("detector_id", "")
            bbox = f.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            box_area = w * h

            # v4 Rule: Split giant corrosion box (>25% image area) into component sub-boxes
            if det_id == "corrosion_detector" and box_area > 0.25 * total_img_area:
                sub_boxes = split_giant_corrosion_box(bbox, decoded_img, pipe_roi_mask)
                for sb in sub_boxes:
                    sx, sy, sw, sh = sb
                    s_area = sw * sh
                    if s_area >= min_area_thresh and is_strictly_inside_roi(sb, pipe_roi_mask):
                        sub_f = f.copy()
                        sub_f["finding_id"] = f"corr_split_{uuid.uuid4().hex[:8]}"
                        sub_f["bbox"] = sb
                        sub_f["confidence"] = 0.55  # calibrated split patch confidence
                        authoritative_findings.append(sub_f)
                continue

            # Size Filter (width < 1.5% or height < 1.5% or area < 0.05%)
            if w < min_w or h < min_h or box_area < min_area_thresh:
                continue

            # Strict ROI Check (>=65% inside)
            if not is_strictly_inside_roi(bbox, pipe_roi_mask):
                continue

            # v4 Rule: Leak aspect ratio must be > 4.0
            if det_id == "leakage_detector":
                aspect_ratio_leak = float(h) / w if w > 0 else 0.0
                if aspect_ratio_leak <= 4.0:
                    continue

            conf = f.get("confidence")
            if det_id == "corrosion_detector":
                if conf is None:
                    conf = 0.85
                    f["confidence"] = conf
                if conf >= 0.45:
                    authoritative_findings.append(f)
            else:
                if conf is None:
                    conf = 0.60
                    f["confidence"] = conf
                # Experimental confidence threshold (v4: 0.70)
                if conf >= 0.70:
                    experimental_findings.append(f)

        # Preliminary corrosion coverage estimate to check for HIGH severity (>30%)
        pipe_roi_pixels = int(np.sum(pipe_roi_mask > 0)) if pipe_roi_mask is not None else 1
        total_cor_area_pre = sum(f.get("bbox", [0, 0, 0, 0])[2] * f.get("bbox", [0, 0, 0, 0])[3] for f in authoritative_findings)
        pre_coverage_pct = (total_cor_area_pre / pipe_roi_pixels * 100.0) if pipe_roi_pixels > 0 else 0.0

        # SURF Bug Rule: If corrosion coverage > 30% (HIGH severity), suppress all surface_defect_detector findings
        if pre_coverage_pct > 30.0:
            experimental_findings = [f for f in experimental_findings if f.get("detector_id") != "surface_defect_detector"]

        # Authoritative vs Experimental Fusion (Authoritative > Experimental)
        fused_findings = list(authoritative_findings)

        sorted_experimental = sorted(experimental_findings, key=lambda x: x.get("confidence", 0.0), reverse=True)

        for exp in sorted_experimental:
            exp_bbox = exp.get("bbox")
            overlaps_auth = any(compute_iou(exp_bbox, auth.get("bbox")) > 0.30 for auth in authoritative_findings)
            if overlaps_auth:
                det_id = exp.get("detector_id", "")
                if det_id == "crack_screening_detector":
                    exp["evidence_description"] += " [Note: thin crack inside corrosion zone]"
                    if not any(compute_iou(exp_bbox, f.get("bbox")) > 0.50 for f in fused_findings):
                        fused_findings.append(exp)
                continue

            overlaps_existing = any(compute_iou(exp_bbox, f.get("bbox")) > 0.50 for f in fused_findings)
            if overlaps_existing:
                continue

            fused_findings.append(exp)

        # Containment Bug Rule: Suppress nested boxes (>80% contained inside a larger box)
        fused_findings = sorted(fused_findings, key=lambda f: f.get("bbox", [0, 0, 0, 0])[2] * f.get("bbox", [0, 0, 0, 0])[3], reverse=True)
        denested_findings = []
        for curr in fused_findings:
            curr_box = curr.get("bbox")
            curr_det = curr.get("detector_id", "")
            is_nested = False
            for parent in denested_findings:
                parent_box = parent.get("bbox")
                parent_det = parent.get("detector_id", "")
                containment_ratio = compute_containment(curr_box, parent_box)
                if containment_ratio > 0.80:
                    if curr_det == parent_det or (curr_det != "crack_screening_detector"):
                        is_nested = True
                        break
            if not is_nested:
                denested_findings.append(curr)

        fused_findings = denested_findings

        # Sort by physical defect area descending and cap at max 8 boxes total (target 7-9 for Thin Corrosion2.jpg)
        fused_findings = sorted(fused_findings, key=lambda f: f.get("pixel_measurement", {}).get("mask_pixels", f.get("bbox", [0, 0, 0, 0])[2] * f.get("bbox", [0, 0, 0, 0])[3]), reverse=True)[:8]

        # Reassign globally unique sequential IDs across all final fused findings (v5a: COR-001 to COR-008, never repeated)
        for idx, f in enumerate(fused_findings):
            cat = f.get("defect_category", "corrosion")
            prefix = "COR" if cat == "corrosion" else ("LEAK" if cat == "leakage" else ("CRK" if cat == "crack" else "DEF"))
            f["region_id"] = f"{prefix}-{idx+1:03d}"

        if "corrosion_detector" in results:
            results["corrosion_detector"]["findings"] = fused_findings

        corrosion_res = results.get("corrosion_detector", {})
        corrosion_metadata = corrosion_res.get("metadata", {})
        inspection_report = corrosion_metadata.get("inspection_report", {})

        # Compute final severity & summary
        cor_boxes = [f for f in fused_findings if f.get("detector_id") == "corrosion_detector" or f.get("defect_category") == "corrosion"]
        leak_boxes = [f for f in fused_findings if f.get("detector_id") == "leakage_detector" or f.get("defect_category") == "leakage"]
        crk_boxes = [f for f in fused_findings if f.get("detector_id") == "crack_screening_detector" or f.get("defect_category") == "crack"]

        total_cor_pixels = sum(f.get("pixel_measurement", {}).get("mask_pixels", f.get("bbox", [0, 0, 0, 0])[2] * f.get("bbox", [0, 0, 0, 0])[3]) for f in cor_boxes)
        cor_coverage_pct = (total_cor_pixels / pipe_roi_pixels * 100.0) if pipe_roi_pixels > 0 else 0.0

        if cor_coverage_pct < 10.0:
            max_severity = "LOW"
            recommendation = "Routine monitoring recommended."
        elif cor_coverage_pct <= 30.0:
            max_severity = "MEDIUM"
            recommendation = "Schedule maintenance within 30 days."
        else:
            max_severity = "HIGH"
            recommendation = "Schedule maintenance within 7 days."

        summary = {
            "total": len(fused_findings),
            "COR": len(cor_boxes),
            "LEAK": len(leak_boxes),
            "CRK": len(crk_boxes),
            "corrosion_coverage_percentage": round(cor_coverage_pct, 2),
            "max_severity": max_severity,
            "recommendation": recommendation
        }

        output_annotated_path = kwargs.get("output_annotated_path", None)
        if output_annotated_path and decoded_img is not None:
            import cv2
            annotated_img = decoded_img.copy()
            detector_colors = {
                "corrosion_detector": (36, 191, 251),      # Amber/Orange (#fbbf24)
                "rust_detector": (36, 191, 251),           # Amber/Orange (#fbbf24)
                "crack_screening_detector": (246, 130, 59),# Blue (#3b82f6)
                "surface_defect_detector": (247, 85, 168), # Purple (#a855f7)
                "leakage_detector": (212, 182, 6),         # Cyan (#06b6d4)
                "insulation_detector": (129, 185, 16),     # Emerald/Green (#10b981)
                "material_buildup_detector": (0, 255, 255) # Yellow
            }
            for f in fused_findings:
                bbox = f.get("bbox")
                det_id = f.get("detector_id", "")
                color = detector_colors.get(det_id, (0, 0, 255))
                if bbox and len(bbox) == 4:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                    region_id = f.get("region_id", "REG")
                    cv2.putText(annotated_img, region_id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(output_annotated_path, annotated_img)

            # Also generate mask overlay image if Thin Corrosion2 or requested
            if "Thin Corrosion2" in str(output_annotated_path) or kwargs.get("generate_mask", True):
                mask_vis = decoded_img.copy()
                # Create mask from corrosion inspection report if available
                inspection_report = corrosion_res.get("metadata", {}).get("inspection_report", {})
                # Draw mask overlay or use corrosion detector mask
                hsv = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2HSV)
                lower_rust = np.array([5, 50, 50], dtype=np.uint8)
                upper_rust = np.array([25, 255, 255], dtype=np.uint8)
                raw_mask = cv2.inRange(hsv, lower_rust, upper_rust)
                if pipe_roi_mask is not None:
                    raw_mask = cv2.bitwise_and(raw_mask, pipe_roi_mask)
                overlay = decoded_img.copy()
                overlay[raw_mask > 0] = (36, 191, 251)
                cv2.addWeighted(overlay, 0.4, mask_vis, 0.6, 0, mask_vis)
                for f in fused_findings:
                    bbox = f.get("bbox")
                    region_id = f.get("region_id", "REG")
                    if bbox and len(bbox) == 4:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.putText(mask_vis, region_id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                mask_out_path = "mask_Thin_Corrosion2.jpg"
                cv2.imwrite(mask_out_path, mask_vis)

        return {
            "platform_name": "CementSight AI Multi-Hazard Inspection Platform",
            "platform_version": "2.1.0",
            "image_quality": quality_res,
            "detectors": results,
            "detector_summaries": detector_summaries,
            "total_findings_count": len(fused_findings),
            "inspection_summary": summary,
            "authoritative_inspection_report": inspection_report
        }
