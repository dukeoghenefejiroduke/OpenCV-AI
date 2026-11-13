"""Multi-Hazard Detector Orchestrator for CementSight AI (vision/orchestrator.py).

Coordinates execution across multiple independent hazard detectors (corrosion, cracks, surface defects,
leakage, insulation, and experimental material buildup), running an image quality gate first and aggregating
results into a unified platform inspection report while maintaining strict fault isolation.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from vision.contract import BaseDetector, validate_finding, STATUS_DETECTOR_FAILED
from vision.quality_gate import ImageQualityGate
from vision.corrosion_detector import CorrosionDetector
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector
from vision.material_buildup_detector import MaterialBuildupDetector


class MultiHazardOrchestrator:
    """Orchestrates multiple modular vision detectors with pre-inspection quality gating and fault isolation."""

    def __init__(self, enable_experimental_buildup: bool = False):
        self.quality_gate = ImageQualityGate()
        self.detectors: List[BaseDetector] = [
            CorrosionDetector(),
            CrackDetector(),
            SurfaceDefectDetector(),
            LeakageDetector(),
            InsulationDetector()
        ]
        if enable_experimental_buildup:
            self.detectors.append(MaterialBuildupDetector())

    def register_detector(self, detector: BaseDetector):
        """Registers a new modular detector at runtime."""
        self.detectors.append(detector)

    def run_inspections(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs image quality assessment, then all registered detectors with fault isolation.

        Args:
            image: BGR numpy array or image path.
            **kwargs: Extra parameters passed to detectors.

        Returns:
            Aggregated multi-hazard inspection report with quality gate status.
        """
        # Run image quality gate first
        # Handle path vs numpy array for quality gate
        if isinstance(image, str):
            import cv2
            decoded_img = cv2.imread(image)
        else:
            decoded_img = image

        quality_res = self.quality_gate.evaluate(decoded_img if decoded_img is not None else np.zeros((100, 100, 3), dtype=np.uint8))
        is_low_quality = (quality_res.get("verdict") == "LOW_QUALITY")

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

            # If image quality is LOW_QUALITY, adjust/cap confidence and tag findings
            findings = res.get("findings", [])
            if is_low_quality:
                for f in findings:
                    f["confidence"] = 0.2  # low confidence due to poor image quality
                    if "limitations" in f and isinstance(f["limitations"], list):
                        f["limitations"].append("Warning: Image quality gate flagged low quality (retake image recommended). Findings have low confidence.")

            # Validate findings in result
            valid_findings = []
            img_shape = decoded_img.shape[:2] if decoded_img is not None else None
            for f in findings:
                is_valid, err = validate_finding(f, image_shape=img_shape)
                if is_valid:
                    valid_findings.append(f)
                else:
                    pass
            res["findings"] = valid_findings
            results[detector.detector_id] = res

            detector_summaries.append({
                "detector_id": detector.detector_id,
                "detector_name": detector.detector_name,
                "version": detector.version,
                "defect_category": detector.defect_category,
                "execution_status": res.get("execution_status"),
                "findings_count": len(valid_findings),
                "errors": res.get("errors", []),
                "limitations": res.get("limitations", [])
            })
            all_findings.extend(valid_findings)

        # Production corrosion result is the authoritative safety/decision baseline
        corrosion_res = results.get("corrosion_detector", {})
        corrosion_metadata = corrosion_res.get("metadata", {})
        inspection_report = corrosion_metadata.get("inspection_report", {})

        return {
            "platform_name": "CementSight AI Multi-Hazard Inspection Platform",
            "platform_version": "2.1.0",
            "image_quality": quality_res,
            "detectors": results,
            "detector_summaries": detector_summaries,
            "total_findings_count": len(all_findings),
            "authoritative_inspection_report": inspection_report
        }
