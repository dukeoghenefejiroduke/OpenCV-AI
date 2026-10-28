"""Unit test suite for CementSight Advanced Features (tests/test_advanced_features.py).

Tests predictive forecasting data sufficiency, spalling/crack detection, CMMS work-order draft formatting,
and stream simulation marking.
"""

import unittest
import numpy as np
import cv2

from app.forecasting import calculate_degradation_velocity, project_time_to_critical
from vision.spalling import detect_spalling_and_cracks
from app.cmms import format_cmms_work_order
from app.digital_twin import AssetNode
from vision.stream import StreamProcessor
from agent.decision import evaluate_inspection


class TestAdvancedFeatures(unittest.TestCase):

    def test_forecasting_velocity_and_projection(self):
        history = [(0.0, 2.0), (10.0, 4.0)]
        velocity = calculate_degradation_velocity(history)
        self.assertAlmostEqual(velocity, 0.2)  # 2% increase in 10 days = 0.2%/day

        days = project_time_to_critical(current_corrosion=6.0, daily_rate=velocity, critical_threshold=10.0)
        self.assertAlmostEqual(days, 20.0)  # (10 - 6) / 0.2 = 20 days

    def test_forecasting_insufficient_data(self):
        # 0 or 1 observation should return 0.0
        self.assertEqual(calculate_degradation_velocity([]), 0.0)
        self.assertEqual(calculate_degradation_velocity([(0.0, 2.0)]), 0.0)
        # Invalid data / non-finite should return 0.0
        self.assertEqual(calculate_degradation_velocity([(0.0, float('nan')), (1.0, 5.0)]), 0.0)

    def test_spalling_detection(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        cv2.line(img, (10, 10), (180, 180), (20, 20, 20), 2)
        cv2.line(img, (20, 150), (150, 20), (20, 20, 20), 2)
        result = detect_spalling_and_cracks(img)
        self.assertIn("crack_count", result)
        self.assertIn("spalling_score", result)

    def test_cmms_work_order_formatting(self):
        report = {"corrosion_percentage": 15.0, "region_count": 5, "recommendation": "CRITICAL"}
        decision = evaluate_inspection(report)
        work_order = format_cmms_work_order("KILN-FEED-PIPE-03", decision, 15.0)

        self.assertEqual(work_order["asset_id"], "KILN-FEED-PIPE-03")
        self.assertEqual(work_order["priority"], "HIGH")
        self.assertTrue(work_order["approval_required"])
        self.assertEqual(work_order["dispatch_status"], "PAYLOAD PREPARED — NOT DISPATCHED")
        self.assertIn("integration_notice", work_order)

    def test_digital_twin_asset_node(self):
        node = AssetNode(zone="Plant Zone A", equipment_type="Rotary Kiln Pipe", asset_id="PIPE-01", criticality="CRITICAL")
        report = {"corrosion_percentage": 4.5, "region_count": 3, "recommendation": "WARNING"}
        node.update_inspection(report, "WARNING")
        data = node.to_dict()
        self.assertEqual(data["zone"], "Plant Zone A")
        self.assertEqual(data["asset_id"], "PIPE-01")
        self.assertEqual(data["last_inspection"]["decision"], "WARNING")

    def test_stream_processor_fallback(self):
        processor = StreamProcessor("non_existent_stream.mp4")
        frames = list(processor.extract_frames(max_frames=1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0], 0)
        self.assertEqual(frames[0][1].shape, (300, 300, 3))
        self.assertTrue(processor.is_simulated)


if __name__ == "__main__":
    unittest.main()
