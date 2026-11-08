# CementSight AI — Pilot Operator & Deployment Guide

## 1. Overview
CementSight AI is an affordable, offline-first infrastructure inspection and maintenance intelligence platform designed for water utilities, industrial facilities, maintenance contractors, and engineering inspection teams operating in resource-constrained environments (such as Android Termux).

---

## 2. Installation & Launch (Android Termux)

1. Ensure Python 3 and required packages are installed:
   ```bash
   pkg update && pkg install python git
   pip install -r requirements.txt
   ```
2. Launch the local offline server:
   ```bash
   python tools/server.py
   ```
3. Open your mobile or desktop browser and navigate to:
   `http://127.0.0.1:8000`

---

## 3. Operator & Inspector Workflow

1. **Asset Registration:** Enter Asset ID, location, material, criticality, and optional QR payload.
2. **Inspection Execution:** Select asset, upload or capture pipe inspection photograph, enter inspector notes, and run detectors.
3. **Finding Review:** Inspect candidate suggestions (corrosion, cracks, surface defects, leakage, insulation). Mark each finding as `confirmed`, `rejected`, or `needs_followup` with review notes.
4. **Maintenance Action:** Record maintenance work performed (e.g., epoxy coating repair).
5. **Repair Verification & Follow-up:** Conduct a follow-up inspection and link it to the repair record as `verified_resolved`.
6. **Report Generation:** Export a self-contained offline HTML inspection report with complete limitation disclaimers.

---

## 4. QR Identification & Fallback Workflow
- **QR Generation & Tagging:** Assets can store a standardized QR URI payload (`CEMENTSIGHT://ASSET/<asset_id>`).
- **Field Scanning Fallback:** Where native camera stream decoding in restricted mobile browsers is unavailable, inspectors can use their phone's built-in camera/QR scanner to read the physical asset QR tag, which opens the direct asset URL in CementSight AI.

---

## 5. Backup, Verification & Recovery Instructions

- **Create a Verified Backup:**
  ```python
  from workspace.backup import create_backup
  backup_dir = create_backup(workspace_root="data/workspace", backup_dest_dir="data/backups")
  print("Backup created at:", backup_dir)
  ```
- **Restore into Isolated Location:**
  ```python
  from workspace.backup import restore_backup
  restore_backup("data/backups/backup_YYYYMMDD_HHMMSS", target_workspace_root="data/workspace_restored")
  ```

---

## 6. Known Limitations
- RGB optical imagery alone cannot establish corrosion depth, remaining wall thickness, internal structural integrity, active leakage, or thermal insulation performance.
- Pixel measurements are not physical dimensions without calibrated references.
- Dataset partitions (Tubers) contain capture session similarities and should not be used for cross-site generalization claims without local calibration.

---

## 7. Pilot Readiness Classification
**Classification:** **Controlled Pilot Candidate**
*(All core features, offline storage, prioritization, repair verification, and automated tests are fully implemented and verified with 110 passing tests. Ready for supervised pilot deployment.)*
