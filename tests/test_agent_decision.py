"""Unit and End-to-End Tests for CementSight Agent Decision Layer (agent/decision.py) & CLI Workflow Integration."""

import os
import cv2
import numpy as np

from agent.decision import evaluate_inspection, inspect_and_decide, InspectionDecision
from vision.inspection import inspect_corrosion


def test_pass_inspection():
    # Mock inspection report with 0.5% corrosion (< 2.0%)
    report = {
        "corrosion_percentage": 0.5,
        "region_count": 0,
        "recommendation": "PASS",
        "dimensions": {"width": 200, "height": 200},
        "bounding_boxes": []
    }
    decision = evaluate_inspection(report)
    assert decision.decision == "PASS"
    assert "normal" in decision.action.lower() or "no maintenance escalation" in decision.action.lower()
    assert decision.approval_required is False
    assert "0.5" in decision.reason
    print("test_pass_inspection PASSED")


def test_warning_inspection():
    # Mock inspection report with 5.0% corrosion (between 2.0% and 10.0%)
    report = {
        "corrosion_percentage": 5.0,
        "region_count": 2,
        "recommendation": "WARNING",
        "dimensions": {"width": 300, "height": 300},
        "bounding_boxes": [(10, 10, 20, 20), (50, 50, 30, 30)]
    }
    decision = evaluate_inspection(report)
    assert decision.decision == "WARNING"
    assert "routine maintenance inspection" in decision.action.lower()
    assert decision.approval_required is False
    assert "5.0" in decision.reason
    print("test_warning_inspection PASSED")


def test_critical_inspection():
    # Mock inspection report with 15.0% corrosion (>= 10.0%)
    report = {
        "corrosion_percentage": 15.0,
        "region_count": 4,
        "recommendation": "CRITICAL",
        "dimensions": {"width": 400, "height": 400},
        "bounding_boxes": [(0, 0, 50, 50)]
    }
    decision = evaluate_inspection(report)
    assert decision.decision == "CRITICAL"
    assert "work-order request" in decision.action.lower() or "work-order" in decision.action.lower()
    assert decision.approval_required is True
    assert "15.0" in decision.reason
    print("test_critical_inspection PASSED")


def test_critical_requires_human_approval():
    # Ensure CRITICAL explicitly requires human approval and does not auto-dispatch
    report = {
        "corrosion_percentage": 12.5,
        "region_count": 3,
        "recommendation": "CRITICAL",
        "dimensions": {"width": 300, "height": 300},
        "bounding_boxes": []
    }
    decision = evaluate_inspection(report)
    assert decision.decision == "CRITICAL"
    assert decision.approval_required is True
    # Verify action / reason clearly states proposed/awaiting, not dispatched
    action_lower = decision.action.lower()
    assert "proposed" in action_lower or "awaiting" in action_lower
    assert "dispatched" not in action_lower
    print("test_critical_requires_human_approval PASSED")


def test_malformed_inspection_data():
    # Test safe handling of None, empty dict, or invalid types
    for malformed_input in [None, {}, "invalid", []]:
        decision = evaluate_inspection(malformed_input)
        assert decision.decision in ["PASS", "WARNING", "CRITICAL"]
        assert isinstance(decision.action, str)
        assert isinstance(decision.reason, str)
        assert isinstance(decision.approval_required, bool)
    print("test_malformed_inspection_data PASSED")


def test_corrosion_percentage_influence():
    # Verify that corrosion percentage directly drives decision thresholds independently
    low_report = {"corrosion_percentage": 1.0, "region_count": 1, "recommendation": "PASS"}
    mid_report = {"corrosion_percentage": 4.5, "region_count": 2, "recommendation": "PASS"} # percentage >= 2.0 triggers WARNING
    high_report = {"corrosion_percentage": 11.0, "region_count": 5, "recommendation": "WARNING"} # percentage >= 10.0 triggers CRITICAL

    assert evaluate_inspection(low_report).decision == "PASS"
    assert evaluate_inspection(mid_report).decision == "WARNING"
    assert evaluate_inspection(high_report).decision == "CRITICAL"
    print("test_corrosion_percentage_influence PASSED")


def test_no_false_external_work_order_dispatch_claims():
    # Verify that action statements never claim an external work order was actually dispatched
    for percentage in [0.5, 4.0, 15.0]:
        report = {"corrosion_percentage": percentage, "region_count": 1}
        decision = evaluate_inspection(report)
        action = decision.action.lower()
        assert "dispatched" not in action
        assert "sent" not in action
        assert "executed" not in action
        assert "proposed" in action or "recorded" in action
    print("test_no_false_external_work_order_dispatch_claims PASSED")


def test_end_to_end_real_image():
    # End-to-end test using data/pipe_corrosion.jpg
    image_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(image_path):
        os.makedirs("data", exist_ok=True)
        dummy_img = np.full((300, 300, 3), 150, dtype=np.uint8)
        cv2.rectangle(dummy_img, (50, 50), (150, 150), (20, 80, 180), -1)
        cv2.imwrite(image_path, dummy_img)

    decision = inspect_and_decide(image_path, output_annotated_path="data/outputs/annotated_pipe_corrosion.jpg")
    
    assert isinstance(decision, InspectionDecision)
    assert decision.decision in ["PASS", "WARNING", "CRITICAL"]
    assert isinstance(decision.action, str)
    assert isinstance(decision.reason, str)
    assert isinstance(decision.approval_required, bool)
    assert "corrosion_percentage" in decision.inspection_summary

    print(f"End-to-end real image result: Decision={decision.decision}, Corrosion%={decision.inspection_summary['corrosion_percentage']}, ApprovalRequired={decision.approval_required}")
    print("test_end_to_end_real_image PASSED")


def test_integrated_workflow_with_real_image():
    # Integration test verifying app workflow: vision inspection -> agent decision evaluation
    image_path = "data/pipe_corrosion.jpg"
    assert os.path.exists(image_path), f"Test image {image_path} must exist."

    # Step 1: Run vision inspection
    inspection_report = inspect_corrosion(image_path, output_annotated_path="data/outputs/annotated_pipe_corrosion.jpg")
    
    assert isinstance(inspection_report, dict)
    assert "corrosion_percentage" in inspection_report
    assert "recommendation" in inspection_report
    assert "region_count" in inspection_report

    # Step 2: Run agent decision evaluation
    decision = evaluate_inspection(inspection_report)

    # Step 3: Assert integration properties
    assert isinstance(decision, InspectionDecision)
    assert decision.decision == inspection_report["recommendation"]
    assert decision.inspection_summary["corrosion_percentage"] == inspection_report["corrosion_percentage"]
    assert decision.inspection_summary["region_count"] == inspection_report["region_count"]

    # For data/pipe_corrosion.jpg, recommendation should be WARNING and action routine maintenance
    assert decision.decision == "WARNING"
    assert "routine maintenance inspection" in decision.action.lower()
    assert decision.approval_required is False

    print(f"test_integrated_workflow_with_real_image PASSED (Corrosion: {inspection_report['corrosion_percentage']}%, Decision: {decision.decision})")


if __name__ == "__main__":
    print("Running Agent Decision Unit & Integration Tests...")
    test_pass_inspection()
    test_warning_inspection()
    test_critical_inspection()
    test_critical_requires_human_approval()
    test_malformed_inspection_data()
    test_corrosion_percentage_influence()
    test_no_false_external_work_order_dispatch_claims()
    test_end_to_end_real_image()
    test_integrated_workflow_with_real_image()
    print("All agent decision & integration tests passed successfully!")
