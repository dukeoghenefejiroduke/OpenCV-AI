"""Live end-to-end verification script for CementSight AI emergency repair.
Tests all workflows empirically on a live background HTTP server instance:
a) Asset CRUD (POST/GET/PUT/DELETE)
b) Image upload & corrosion inspection with bounding boxes
c) Finding review (confirm/reject) & persistence across server restart
d) Telemetry CSV upload & MAD Z-score/EWMA/CUSUM anomaly detection
e) Digital twin loading & risk fusion
f) Maintenance work order creation, status update, & persistence
g) HTML report generation & CSV export
h) Backup creation & restoration
"""

import os
import sys
import time
import json
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.server import ReuseTCPServer, CementSightHandler


def run_live_verification():
    port = 8989
    server = ReuseTCPServer(("127.0.0.1", port), CementSightHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{port}"
    print(f"=== Starting Live End-to-End Verification on {base_url} ===")

    try:
        # 1. Asset CRUD (POST, GET, PUT, DELETE)
        print("[Test 1] Asset CRUD (POST/GET/PUT/DELETE)...")
        # POST asset
        asset_payload = json.dumps({
            "asset_id": "TEST-PIPE-99",
            "location": "Verification Plant Zone",
            "pipe_material": "steel",
            "criticality": "high",
            "notes": "Initial creation"
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/workspace/assets", data=asset_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res_data = json.loads(resp.read().decode())
            assert res_data["status"] == "ok"
        print(" -> Asset POST passed.")

        # GET asset
        req = urllib.request.urlopen(f"{base_url}/api/workspace/asset/TEST-PIPE-99")
        assert req.status == 200
        get_data = json.loads(req.read().decode())
        assert get_data["asset"]["criticality"] == "high"
        print(" -> Asset GET passed.")

        # PUT asset (Update)
        put_payload = json.dumps({"criticality": "critical", "notes": "Updated criticality"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/workspace/asset/TEST-PIPE-99", data=put_payload, headers={"Content-Type": "application/json"}, method="PUT")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        print(" -> Asset PUT passed.")

        # 2. Image upload & inspection inspection
        print("[Test 2] Image upload & inspection inspection...")
        img_path = "data/pipe_corrosion.jpg"
        assert os.path.exists(img_path)
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Build multipart/form-data payload manually
        boundary = "----CementSightBoundary123456"
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"asset_id\"\r\n\r\nTEST-PIPE-99\r\n"
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image\"; filename=\"pipe_corrosion.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/workspace/inspection",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 201
            insp_data = json.loads(resp.read().decode())
            insp_id = insp_data["inspection_id"]
            assert os.path.exists(insp_data["annotated_output"])
        print(f" -> Inspection POST & Bounding-Box Overlay passed (ID: {insp_id}).")

        # 3. Confirming and rejecting a finding & persistence
        print("[Test 3] Finding review & persistence across restart...")
        # Get inspection details to find findings
        req = urllib.request.urlopen(f"{base_url}/api/workspace/inspection/{insp_id}")
        assert req.status == 200
        insp_detail = json.loads(req.read().decode())
        findings = insp_detail.get("findings", [])
        if findings:
            finding_id = findings[0]["finding_id"]
            review_payload = json.dumps({"status": "confirmed", "notes": "Verified verified corrosion"}).encode("utf-8")
            req = urllib.request.Request(f"{base_url}/api/workspace/review/{finding_id}", data=review_payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
            print(" -> Finding review (confirmed) passed.")

        # Shutdown server and restart to verify persistence in SQLite
        server.shutdown()
        server.server_close()
        t.join()

        # Restart server
        server = ReuseTCPServer(("127.0.0.1", port), CementSightHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.5)

        # Verify inspection and findings still exist
        req = urllib.request.urlopen(f"{base_url}/api/workspace/inspection/{insp_id}")
        assert req.status == 200
        persisted_detail = json.loads(req.read().decode())
        assert len(persisted_detail["findings"]) > 0
        if findings:
            assert persisted_detail["findings"][0]["human_review_status"] == "confirmed"
        print(" -> SQLite persistence across server restart passed.")

        # 4. Telemetry CSV upload & MAD Z-score / EWMA / CUSUM anomaly detection
        print("[Test 4] Telemetry CSV upload & anomaly detection...")
        csv_data = "timestamp,vibration\n2026-11-18 10:00:00,1.2\n2026-11-18 10:01:00,1.3\n2026-11-18 10:02:00,9.8\n2026-11-18 10:03:00,1.4\n"
        tel_payload = json.dumps({"asset_id": "TEST-PIPE-99", "csv_content": csv_data}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/v2/telemetry/upload", data=tel_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            tel_res = json.loads(resp.read().decode())
            assert tel_res["status"] == "success"
            assert tel_res["anomalies_detected"] > 0
        print(" -> Telemetry CSV upload & anomaly detection passed.")

        # 5. Digital Twin & Risk Fusion
        print("[Test 5] Digital Twin & Risk Fusion...")
        req = urllib.request.urlopen(f"{base_url}/api/v2/twin/network")
        assert req.status == 200
        twin_data = json.loads(req.read().decode())
        assert twin_data["badge"] == "[SIMULATED]"

        risk_payload = json.dumps({
            "visual_report": {"corrosion_percentage": 25.0, "recommendation": "CRITICAL", "confirmed": True},
            "anomalies": [{"detector": "robust_zscore", "score": 4.5}],
            "provenance": "real"
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/v2/risk/fuse", data=risk_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            risk_res = json.loads(resp.read().decode())
            assert risk_res["risk_assessment"]["risk_level"] == "CRITICAL"
        print(" -> Digital Twin & Risk Fusion passed.")

        # 6. Maintenance Work Order & Status Update
        print("[Test 6] Maintenance Work Order & Status Update...")
        maint_payload = json.dumps({
            "asset_id": "TEST-PIPE-99",
            "inspection_id": insp_id,
            "work_performed": "Replaced corroded valve section",
            "performed_by": "Lead Tech J. Doe",
            "kind": "recorded_work"
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/workspace/maintenance", data=maint_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            maint_res = json.loads(resp.read().decode())
            record_id = maint_res["record_id"]

        # Update maintenance status (PUT)
        status_payload = json.dumps({"verification_status": "verified_resolved", "followup_inspection_id": insp_id}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/workspace/maintenance/{record_id}", data=status_payload, headers={"Content-Type": "application/json"}, method="PUT")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        print(" -> Maintenance work order creation & status update passed.")

        # 7. HTML Report & CSV Export
        print("[Test 7] HTML Report & CSV Export...")
        req = urllib.request.urlopen(f"{base_url}/api/workspace/report/{insp_id}")
        assert req.status == 200
        assert "<html" in req.read().decode().lower()

        req = urllib.request.urlopen(f"{base_url}/api/workspace/export-csv")
        assert req.status == 200
        assert "inspection_id" in req.read().decode()

        req = urllib.request.urlopen(f"{base_url}/api/workspace/asset-export-csv/TEST-PIPE-99")
        assert req.status == 200
        assert "TEST-PIPE-99" in req.read().decode()
        print(" -> HTML Report & CSV export passed.")

        # 8. Backup & Restoration
        print("[Test 8] Backup & Restoration...")
        req = urllib.request.Request(f"{base_url}/api/workspace/backup", method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            backup_res = json.loads(resp.read().decode())
            backup_dir = backup_res["backup_dir"]
            assert os.path.exists(backup_dir)

        # Restore backup
        restore_payload = json.dumps({"backup_dir": backup_dir}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/workspace/restore", data=restore_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            restore_res = json.loads(resp.read().decode())
            assert restore_res["status"] == "restored"
        print(" -> Backup creation & restoration passed.")

        # Clean up asset (DELETE)
        req = urllib.request.Request(f"{base_url}/api/workspace/asset/TEST-PIPE-99", method="DELETE")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200

        print("=== ALL LIVE VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

    finally:
        server.shutdown()
        server.server_close()
        t.join()


if __name__ == "__main__":
    run_live_verification()
