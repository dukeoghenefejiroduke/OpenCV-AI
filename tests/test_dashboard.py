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

    def test_7_api_inspect_absolute_path_rejected(self):
        url = f"http://127.0.0.1:{self.port}/api/inspect"
        abs_path = os.path.abspath("data/pipe_corrosion.jpg")
        payload = json.dumps({"image_path": abs_path, "asset_id": "PIPE-01"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_8_api_v2_twin_network(self):
        url = f"http://127.0.0.1:{self.port}/api/v2/twin/network"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data.get("badge"), "[SIMULATED]")
        self.assertIn("health_aggregation", data)

    def test_9_api_v2_telemetry_simulate_and_get(self):
        sim_url = f"http://127.0.0.1:{self.port}/api/v2/telemetry/simulate"
        payload = json.dumps({"seed": 42, "asset_id": "SIM-PIPE-TEST", "fault_type": "leak", "n_steps": 20}).encode("utf-8")
        req = urllib.request.Request(sim_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("badge"), "[SIMULATED]")
            self.assertEqual(data.get("status"), "success")

        # Now get telemetry for asset
        get_url = f"http://127.0.0.1:{self.port}/api/v2/telemetry/SIM-PIPE-TEST"
        req_get = urllib.request.urlopen(get_url)
        self.assertEqual(req_get.status, 200)
        get_data = json.loads(req_get.read().decode("utf-8"))
        self.assertEqual(get_data.get("badge"), "[SIMULATED]")
        self.assertGreaterEqual(get_data.get("telemetry_count"), 20)

    def test_10_api_v2_risk_fuse(self):
        url = f"http://127.0.0.1:{self.port}/api/v2/risk/fuse"
        payload = json.dumps({
            "visual_report": {"corrosion_percentage": 18.5, "recommendation": "CRITICAL", "confirmed": True},
            "anomalies": [{"detector": "ewma", "provenance": "simulated"}],
            "provenance": "simulated"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("badge"), "[SIMULATED]")
            self.assertEqual(data["risk_assessment"]["risk_level"], "CRITICAL")

    def test_11_api_workspace_inspection_multi_hazard_detectors(self):
        url = f"http://127.0.0.1:{self.port}/api/workspace/inspection"
        payload = json.dumps({"image_path": "data/pipe_corrosion.jpg", "asset_id": "PIPE-DEF"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            summaries = data.get("detector_summaries", [])
            detector_ids = [s.get("detector_id") for s in summaries]
            # Should run all multi-hazard detectors (enable_experimental=True)
            self.assertIn("corrosion_detector", detector_ids)
            self.assertIn("rust_detector", detector_ids)
            self.assertIn("crack_screening_detector", detector_ids)

    def test_12_api_workspace_backup_and_restore(self):
        # Test backup
        backup_url = f"http://127.0.0.1:{self.port}/api/workspace/backup"
        req_backup = urllib.request.Request(backup_url, method="POST")
        backup_dir = None
        with urllib.request.urlopen(req_backup) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            backup_dir = data.get("backup_dir")
            self.assertIsNotNone(backup_dir)

        # Test restore
        restore_url = f"http://127.0.0.1:{self.port}/api/workspace/restore"
        payload = json.dumps({"backup_dir": backup_dir}).encode("utf-8")
        req_restore = urllib.request.Request(restore_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req_restore) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "restored")


if __name__ == "__main__":
    unittest.main()
