# CementSight AI — Termux Edge Runbook (v2.3)

## 1. Overview
CementSight AI runs as a single-inspector, offline-first edge tool on Android via Termux. It provides classical computer vision corrosion and hazard inspection, asset management, telemetry analysis, digital twin mapping, risk fusion, and compliance reporting.

---

## 2. Quick Start
Open Termux and run:
```bash
bash start.sh
```
This script verifies the workspace database, ensures directories exist, and launches the local server at **`http://127.0.0.1:8989`**.

---

## 3. Database Backup & Disaster Recovery
SQLite operates locally in `workspace/workspace.db`. To prevent data loss:

### Automated Backup
Trigger a backup via the dashboard (Section 8) or via API:
```bash
curl -X POST http://127.0.0.1:8989/api/workspace/backup
```
Backups are saved to `data/backups/`.

### Disaster Recovery (Restore in < 5 mins)
If the database is corrupted or lost:
1. Stop the server (`Ctrl+C`).
2. Run the restore utility or API:
   ```bash
   curl -X POST http://127.0.0.1:8989/api/workspace/restore -H "Content-Type: application/json" -d '{"backup_dir": "data/backups/backup_TIMESTAMP"}'
   ```
3. Restart `bash start.sh`.

---

## 4. Field Operation Guidelines
- **Offline Mode:** The application is fully functional offline via service worker caching.
- **Image Capture:** Keep the camera steady and perpendicular to the asset surface to minimize severe glare or extreme shadows.
- **Human Verification:** **All AI findings require inspector review.** AI detection is an assistive screening tool; the final compliance verdict rests solely with the certified inspector.

---

## 5. Security Boundary
- Bound strictly to **`127.0.0.1:8989`**. Never expose to external networks (`0.0.0.0`) without a secure VPN or reverse proxy with authentication.
