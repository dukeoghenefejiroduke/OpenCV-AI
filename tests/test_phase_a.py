"""Unit tests for Phase A gap closures:
- Multilingual string tables (Pidgin & French, disabled, needs native review)
- Asset tracking comparable observations (>=3 comparable observations, range with 'assumes similar framing')
- Explainability panel derived from real stage counts (6 pipeline stages + 7 diagnostic stages)
"""

import unittest
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.strings import get_string, STRING_TABLES
from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection


class TestPhaseAGaps(unittest.TestCase):

    def test_multilingual_strings_disabled(self):
        # Pidgin and French strings should be disabled by default and return default/empty unless enabled
        self.assertEqual(STRING_TABLES["pcm"]["status"], "disabled")
        self.assertEqual(STRING_TABLES["pcm"]["review_status"], "needs native review")
        self.assertEqual(get_string("pcm", "corrosion_screening", "fallback"), "fallback")

        self.assertEqual(STRING_TABLES["fr"]["status"], "disabled")
        self.assertEqual(STRING_TABLES["fr"]["review_status"], "needs native review")
        self.assertEqual(get_string("fr", "recommendation", "fallback"), "fallback")

    def test_asset_tracking_comparable_observations_and_framing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            asset = Asset(asset_id="pipe-101", location="Test Unit", pipe_material="carbon_steel", pipe_type="pipe")
            store.upsert_asset(asset)

            # Add 3 inspections with same_position=True
            for i in range(3):
                insp = Inspection(
                    inspection_id=f"insp-{i}",
                    asset_id="pipe-101",
                    inspection_date=f"2026-11-{10+i}",
                    image_sha256=f"sha{i}" * 16,
                    image_ext="jpg",
                    detector_results={"same_position": True},
                    authoritative_report={"corrosion_percentage": 5.0 + i},
                    created_at=f"2026-11-{10+i}T00:00:00Z"
                )
                store.create_inspection(insp)

            trend = store.get_asset_corrosion_trend("pipe-101")
            self.assertEqual(trend["comparable_count"], 3)
            self.assertTrue(trend["has_min_observations"])
            self.assertEqual(trend["corrosion_range"], [5.0, 7.0])
            self.assertEqual(trend["corrosion_delta"], 2.0)
            self.assertEqual(trend["framing_assumption"], "assumes similar framing")
            store.close()

    def test_explainability_panel_real_stage_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            panel = store.generate_explainability_panel(authoritative_report={"decision": "WARNING", "corrosion_percentage": 6.56, "reason": "Test reason"})
            
            self.assertEqual(panel["pipeline_stage_count"], 6)
            self.assertEqual(len(panel["pipeline_stages"]), 6)
            self.assertEqual(panel["diagnostic_stages_count"], 7)
            self.assertEqual(panel["framing_assumption"], "assumes similar framing")
            self.assertEqual(panel["decision_rationale"], "Test reason")
            self.assertEqual(panel["corrosion_percentage"], 6.56)
            store.close()
