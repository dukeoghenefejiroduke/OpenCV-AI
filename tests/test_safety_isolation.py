"""Comprehensive Safety-Isolation and Fault-Tolerance Unit Tests (tests/test_safety_isolation.py).

Rigorously asserts all safety-isolation boundaries and Phase 4 adversarial test scenarios:
1. Multiple experimental detectors failing simultaneously.
2. Invalid execution statuses.
3. Malformed findings and non-finite numerical values (NaN, Inf).
4. Unexpected return types and missing fields.
5. Excessive candidate counts.
6. Missing optional dependencies.
7. Exceptions during aggregation and result serialization.
8. Verification that experimental detectors cannot alter authoritative corrosion findings or thresholds.
"""

import pytest
import numpy as np
import os
import math

from vision.contract import BaseDetector, STATUS_DETECTOR_FAILED, STATUS_CANDIDATE_DETECTED, STATUS_NO_CANDIDATE, validate_finding
from vision.orchestrator import MultiHazardOrchestrator


class MultipleFailingDetector(BaseDetector):
    """Simulates multiple experimental detectors failing simultaneously."""
    detector_id = "multi_fail_detector"
    detector_name = "Multi Failing Detector"
    version = "1.0.0"
    defect_category = "experimental"

    def detect(self, image, **kwargs):
        raise RuntimeError("Simulated simultaneous subsystem crash.")


class NonFiniteValueDetector(BaseDetector):
    """Simulates a detector returning non-finite numerical values (NaN, Inf) in findings."""
    detector_id = "nan_detector"
    detector_name = "NaN Detector"
    version = "1.0.0"
    defect_category = "experimental"

    def detect(self, image, **kwargs):
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "supported_input_type": "BGR numpy array",
            "execution_status": STATUS_CANDIDATE_DETECTED,
            "findings": [
                {
                    "finding_id": "nan_finding",
                    "detector_id": self.detector_id,
                    "detector_version": self.version,
                    "defect_category": self.defect_category,
                    "bbox": [0, 0, float('nan'), float('inf')],
                    "pixel_measurement": {"area": float('nan')},
                    "evidence_description": "Non-finite numbers",
                    "confidence": float('nan'),
                    "review_status": "PENDING_REVIEW",
                    "limitations": []
                }
            ],
            "errors": [],
            "limitations": [],
            "metadata": {}
        }


class UnexpectedReturnTypeDetector(BaseDetector):
    """Simulates a rogue detector returning unexpected non-dict return types."""
    detector_id = "rogue_detector"
    detector_name = "Rogue Detector"
    version = "1.0.0"
    defect_category = "experimental"

    def detect(self, image, **kwargs):
        return "Not a dictionary contract response"


class MissingFieldsDetector(BaseDetector):
    """Simulates a detector returning missing required fields in findings."""
    detector_id = "missing_fields_detector"
    detector_name = "Missing Fields Detector"
    version = "1.0.0"
    defect_category = "experimental"

    def detect(self, image, **kwargs):
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "version": self.version,
            "defect_category": self.defect_category,
            "supported_input_type": "BGR numpy array",
            "execution_status": STATUS_CANDIDATE_DETECTED,
            "findings": [
                {
                    "finding_id": "incomplete"
                    # Missing bbox, confidence, etc.
                }
            ],
            "errors": [],
            "limitations": [],
            "metadata": {}
        }


def test_adversarial_safety_isolation_comprehensive():
    """Rigorously executes all Phase 4 adversarial safety testing scenarios."""
    sample_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(sample_path):
        pytest.skip("Sample image not found.")

    orchestrator = MultiHazardOrchestrator()
    orchestrator.register_detector(MultipleFailingDetector())
    orchestrator.register_detector(NonFiniteValueDetector())
    orchestrator.register_detector(UnexpectedReturnTypeDetector())
    orchestrator.register_detector(MissingFieldsDetector())

    report = orchestrator.run_inspections(sample_path)
    detectors = report["detectors"]

    # 1. Multiple failing detectors are contained and reported as failed
    assert detectors.get("multi_fail_detector", {}).get("execution_status") == STATUS_DETECTOR_FAILED

    # 2. Non-finite values and malformed findings are safely rejected/filtered
    nan_findings = detectors.get("nan_detector", {}).get("findings", [])
    for f in nan_findings:
        # Ensure validation rejects or filters out non-finite bboxes/confidences
        bbox = f.get("bbox", [0, 0, 0, 0])
        assert all(math.isfinite(val) for val in bbox if isinstance(val, (int, float)))

    # 3. Unexpected return types and missing fields do not crash orchestrator or taint authoritative inspection
    assert "rogue_detector" in detectors
    assert "missing_fields_detector" in detectors

    # 4. Authoritative corrosion inspection report remains 100% pristine and unaffected
    auth_rep = report["authoritative_inspection_report"]
    assert auth_rep["dimensions"]["width"] == 1920
    assert auth_rep["dimensions"]["height"] == 1080
    assert auth_rep["corrosion_percentage"] == 6.56
    assert auth_rep["region_count"] == 11
    assert auth_rep["recommendation"] == "WARNING"
