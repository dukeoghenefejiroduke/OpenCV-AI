"""Unit test suite for Surface Scoping (Step 0 fix-up).

Tests brick_concrete_wood, not_sure, missing surface field defaulting to not_sure,
storage in inspection record, and appearance in HTML report.
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


class TestSurfaceScoping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = 8998
        cls.server = ReuseTCPServer(("127.0.0.1", cls.port), CementSightHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_brick_concrete_wood_surface(self):
        url = f"http://127.0.0.1:{self.port}/api/workspace/inspection"
        payload = json.dumps({
            "image_path": "data/pipe_corrosion.jpg",
            "asset_id": "PIPE-SURF-1",
            "surface": "brick_concrete_wood"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            auth = data.get("authoritative_inspection_report", {})
            self.assertEqual(auth.get("surface"), "brick_concrete_wood")
            self.assertEqual(auth.get("corrosion_percentage"), 0.0)
            self.assertEqual(auth.get("region_count"), 0)
            self.assertIn("not applicable", auth.get("recommendation", "").lower())

    def test_not_sure_surface(self):
        url = f"http://127.0.0.1:{self.port}/api/workspace/inspection"
        payload = json.dumps({
            "image_path": "data/pipe_corrosion.jpg",
            "asset_id": "PIPE-SURF-2",
            "surface": "not_sure"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            auth = data.get("authoritative_inspection_report", {})
            self.assertEqual(auth.get("surface"), "not_sure")
            self.assertIn("warning", auth.get("reason", "").lower())

    def test_missing_surface_defaults_to_not_sure(self):
        url = f"http://127.0.0.1:{self.port}/api/workspace/inspection"
        payload = json.dumps({
            "image_path": "data/pipe_corrosion.jpg",
            "asset_id": "PIPE-SURF-3"
            # surface omitted intentionally
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            auth = data.get("authoritative_inspection_report", {})
            self.assertEqual(auth.get("surface"), "not_sure")
            self.assertIn("warning", auth.get("reason", "").lower())

    def test_surface_stored_and_in_html_report(self, tmp_path_factory=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = WorkspaceStore(root=tmp_dir)
            asset = Asset(asset_id="PIPE-HTML-1", location="Test Plant")
            store.upsert_asset(asset)
            with open("data/pipe_corrosion.jpg", "rb") as f:
                img_bytes = f.read()
            sha, ext = store.store_image(img_bytes)
            insp = Inspection(
                inspection_id="INS-HTML-1",
                asset_id="PIPE-HTML-1",
                inspection_date="2026-11-15",
                image_sha256=sha,
                image_ext=ext,
                authoritative_report={
                    "corrosion_percentage": 5.0,
                    "surface": "brick_concrete_wood",
                    "recommendation": "Corrosion screening is not applicable to this surface."
                }
            )
            store.create_inspection(insp)
            findings = store.list_findings("INS-HTML-1")
            history = store.list_inspections_for_asset("PIPE-HTML-1")
            p = prioritize(asset, insp, findings, history)
            out_path = os.path.join(tmp_dir, "report.html")
            generate_html_report(store, insp, findings, p, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("brick_concrete_wood", content)
