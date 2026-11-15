"""Unit test suite for non-steel surface option (other_metal_non_steel) and zero-candidate report verdict wording.
"""

import unittest
import os
import sys
import json
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.server import ReuseTCPServer, CementSightHandler
from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection
from workspace.report import generate_html_report
from workspace.prioritization import score as prioritize


class TestNonSteelSurfaceAndVerdict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = 8997
        cls.server = ReuseTCPServer(("127.0.0.1", cls.port), CementSightHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_other_metal_non_steel_surface(self):
        url = f"http://127.0.0.1:{self.port}/api/workspace/inspection"
        payload = json.dumps({
            "image_path": "data/pipe_corrosion.jpg",
            "asset_id": "PIPE-NONSTEEL-1",
            "surface": "other_metal_non_steel"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            auth = data.get("authoritative_inspection_report", {})
            self.assertEqual(auth.get("surface"), "other_metal_non_steel")
            self.assertEqual(auth.get("corrosion_percentage"), 0.0)
            self.assertEqual(auth.get("region_count"), 0)
            self.assertIn("copper, brass, aluminium", auth.get("recommendation", "").lower())
            self.assertIn("not applicable", auth.get("recommendation", "").lower())

    def test_zero_candidate_verdict_wording_in_html_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = WorkspaceStore(root=tmp_dir)
            asset = Asset(asset_id="PIPE-VERDICT-1", location="Test Plant")
            store.upsert_asset(asset)
            with open("data/pipe_corrosion.jpg", "rb") as f:
                img_bytes = f.read()
            sha, ext = store.store_image(img_bytes)
            insp = Inspection(
                inspection_id="INS-VERDICT-1",
                asset_id="PIPE-VERDICT-1",
                inspection_date="2026-11-15",
                image_sha256=sha,
                image_ext=ext,
                authoritative_report={
                    "corrosion_percentage": 0.0,
                    "region_count": 0,
                    "surface": "other_metal_non_steel",
                    "recommendation": "Corrosion screening is not applicable to this surface."
                }
            )
            store.create_inspection(insp)
            findings = []
            history = store.list_inspections_for_asset("PIPE-VERDICT-1")
            p = prioritize(asset, insp, findings, history)
            out_path = os.path.join(tmp_dir, "report.html")
            generate_html_report(store, insp, findings, p, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("No corrosion candidates found. This does not certify the asset.", content)
                self.assertNotIn(">PASS<", content)
