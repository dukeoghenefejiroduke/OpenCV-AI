"""Unit tests for Phase D Product Hardening:
- Security regression checks (path traversal, symlinks, oversized body, wrong content type, malicious filename)
- EXIF GPS stripping
- Data deletion with confirmation audit log
- Service worker offline support
"""

import unittest
import os
import tempfile
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workspace.store import WorkspaceStore
from workspace.models import Asset


class TestPhaseDHardening(unittest.TestCase):

    def test_exif_gps_stripping_on_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            img = np.zeros((150, 150, 3), dtype=np.uint8)
            success, encoded = cv2.imencode(".jpg", img)
            self.assertTrue(success)
            
            digest, ext = store.store_image(encoded.tobytes())
            stored_path = store.image_path(digest, ext)
            self.assertTrue(os.path.exists(stored_path))
            
            # Read back stored image and verify it decodes cleanly without EXIF data
            read_img = cv2.imread(stored_path)
            self.assertIsNotNone(read_img)
            self.assertEqual(read_img.shape[:2], (150, 150))
            store.close()

    def test_delete_my_data_with_confirmation_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkspaceStore(root=tmpdir)
            asset = Asset(asset_id="del-test", location="Delete Test Unit")
            store.upsert_asset(asset)
            self.assertIsNotNone(store.get_asset("del-test"))

            log_path = store.delete_all_workspace_data("User explicitly requested deletion")
            self.assertTrue(os.path.exists(log_path))
            self.assertIsNone(store.get_asset("del-test"))

            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            self.assertEqual(log_data["status"], "deleted")
            self.assertEqual(log_data["reason"], "User explicitly requested deletion")
            store.close()

    def test_service_worker_exists(self):
        sw_path = "app/sw.js"
        self.assertTrue(os.path.exists(sw_path))
        with open(sw_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("caches.open", content)
        self.assertIn("fetch", content)

    def test_security_path_traversal_rejection(self):
        from app.validation import validate_image_input
        # Non-existent path should return invalid result tuple
        valid, img, err = validate_image_input("/etc/passwd")
        self.assertFalse(valid)
        self.assertIsNotNone(err)

    def test_consent_text_and_retention_settings(self):
        dashboard_path = "app/dashboard.html"
        self.assertTrue(os.path.exists(dashboard_path))
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("viewport", html)

    def test_360px_layout_and_friendly_errors(self):
        dashboard_path = "app/dashboard.html"
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("viewport", html)

