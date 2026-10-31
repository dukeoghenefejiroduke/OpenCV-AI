"""Multi-Hazard Detector Orchestrator for CementSight AI (vision/orchestrator.py).

Coordinates execution across multiple independent hazard detectors (corrosion, cracks, etc.),
aggregating results into a unified platform inspection response while maintaining strict separation
between production corrosion rules and experimental screening modules.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from vision.contract import BaseDetector, validate_finding, STATUS_DETECTOR_FAILED
from vision.corrosion_detector import CorrosionDetector
from vision.crack_detector import CrackDetector
from vision.surface_defect_detector import SurfaceDefectDetector
from vision.leakage_detector import LeakageDetector
from vision.insulation_detector import InsulationDetector


class MultiHazardOrchestrator:
    """Orchestrates multiple modular vision detectors with strict fault isolation."""

    def __init__(self):
        self.detectors: List[BaseDetector] = [
            CorrosionDetector(),
            CrackDetector(),
            SurfaceDefectDetector(),
            LeakageDetector(),
            InsulationDetector()
        ]

    def register_detector(self, detector: BaseDetector):
        """Registers a new modular detector at runtime."""
        self.detectors.append(detector)

    def run_inspections(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Runs all registered detectors on the input image with fault isolation.

        Args:
            image: BGR numpy array or image path.
            **kwargs: Extra parameters passed to detectors.

        Returns:
            Aggregated multi-hazard inspection report.
        """
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

            # Validate findings in result
            valid_findings = []
            img_shape = image.shape[:2] if isinstance(image, np.ndarray) else None
            for f in res.get("findings", []):
                is_valid, err = validate_finding(f, image_shape=img_shape)
                if is_valid:
                    valid_findings.append(f)
                else:
                    # Invalid or malformed findings are filtered out for safety
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
            "platform_version": "2.0.0",
            "detectors": results,
            "detector_summaries": detector_summaries,
            "total_findings_count": len(all_findings),
            "authoritative_inspection_report": inspection_report
        }
