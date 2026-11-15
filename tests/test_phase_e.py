"""Unit tests for Phase E Workflow Polish:
- CSV export of inspection history and per-asset history
- Supervisor report print-ready with blank signature and date lines
"""

import unittest
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, PriorityScore
from workspace.report import generate_html_report
from tests.test_workspace_store import JPEG_1PX


class TestPhaseEWorkflowPolish(unittest.TestCase):

    def test_csv_export_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            asset = Asset(asset_id="asset-e", location="Polish Test")
            store.upsert_asset(asset)
            sha, ext = store.store_image(JPEG_1PX)
            
            insp = Inspection(
                inspection_id="insp-e1",
                asset_id="asset-e",
                inspection_date="2026-11-15",
                image_sha256=sha,
                image_ext=ext,
                authoritative_report={"corrosion_percentage": 6.56, "decision": "WARNING"},
                inspector_notes="Test note"
            )
            store.create_inspection(insp)

            csv_global = store.export_inspection_history_csv()
            self.assertIn("insp-e1", csv_global)
            self.assertIn("6.56", csv_global)

            csv_asset = store.export_asset_history_csv("asset-e")
            self.assertIn("insp-e1", csv_asset)

            store.close()

    def test_supervisor_report_blank_signature_and_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            asset = Asset(asset_id="asset-e", location="Polish Test")
            store.upsert_asset(asset)
            sha, ext = store.store_image(JPEG_1PX)
            
            insp = Inspection(
                inspection_id="insp-e1",
                asset_id="asset-e",
                inspection_date="2026-11-15",
                image_sha256=sha,
                image_ext=ext,
                authoritative_report={"corrosion_percentage": 6.56, "decision": "WARNING"}
            )
            store.create_inspection(insp)

            out_html = os.path.join(tmpdir, "report.html")
            priority = PriorityScore(category="routine_review", score=50, rationale=["Test rule"], contributing_factors={})
            generate_html_report(store, insp, [], priority, out_html)

            self.assertTrue(os.path.exists(out_html))
            with open(out_html, "r", encoding="utf-8") as f:
                html_content = f.read()

            self.assertIn("Supervisor Signature:", html_content)
            self.assertIn("___________________________", html_content)
            self.assertIn("Date:", html_content)

            store.close()
