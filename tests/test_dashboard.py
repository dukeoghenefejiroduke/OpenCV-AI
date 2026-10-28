"""Unit test suite for CementSight Dashboard Server & API (tests/test_dashboard.py).

Tests dashboard HTTP server responses, API endpoints (/api/status, /api/report, /api/history, /api/inspect),
input validation, error handling, path traversal protection, and agent decision integration.
"""

import unittest
import os
import sys
import json
import threading
import urllib.request
import urllib.error

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.server import ReuseTCPServer, CementSightHandler, run_server


class TestDashboardServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = 8999
        cls.server = ReuseTCPServer(("127.0.0.1", cls.port), CementSightHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_1_serve_dashboard_index(self):
        url = f"http://127.0.0.1:{self.port}/"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        html_content = req.read().decode("utf-8")
        self.assertIn("CementSight AI", html_content)
        self.assertIn("6-Stage Execution Pipeline", html_content)

    def test_2_api_status(self):
        url = f"http://127.0.0.1:{self.port}/api/status"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data["status"], "ACTIVE")
        self.assertIn("python_version", data)

    def test_3_api_history_and_clear(self):
        # First clear history
        req = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/api/clear-history", method="POST"))
        self.assertEqual(req.status, 200)

        # Check history is empty
        req_hist = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/history")
        self.assertEqual(req_hist.status, 200)
        history = json.loads(req_hist.read().decode("utf-8"))
        self.assertEqual(len(history), 0)

    def test_4_api_inspect_valid_image(self):
        url = f"http://127.0.0.1:{self.port}/api/inspect"
        payload = json.dumps({"image_path": "data/pipe_corrosion.jpg", "asset_id": "PIPE-01"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("stage_3_inspection", data)
                self.assertIn("stage_4_agent_decision", data)
                self.assertEqual(data["stage_4_agent_decision"]["decision"], "WARNING")
        except urllib.error.HTTPError as e:
            self.fail(f"API inspect failed with HTTP error: {e.code} - {e.read().decode()}")

    def test_5_api_inspect_invalid_image(self):
        url = f"http://127.0.0.1:{self.port}/api/inspect"
        payload = json.dumps({"image_path": "non_existent_image.jpg", "asset_id": "PIPE-01"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_6_path_traversal_protection(self):
        # Attempt to access unauthorized file via static server GET
        url = f"http://127.0.0.1:{self.port}/data/../requirements.txt"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url)
        self.assertIn(ctx.exception.code, [403, 404])


if __name__ == "__main__":
    unittest.main()
