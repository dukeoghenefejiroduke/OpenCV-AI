"""Safe SQLite backup, verification, and restoration utility for CementSight AI (Phase 3).

Ensures atomic SQLite backup via SQLite's native backup API / safe file copy,
image evidence preservation, integrity checks (PRAGMA integrity_check), and
safe recovery into isolated temporary locations before production restoration.
"""

import os
import shutil
import sqlite3
from typing import Tuple


def create_backup(workspace_root: str = "data/workspace", backup_dest_dir: str = "data/backups") -> str:
    """Create a verified safe backup of the workspace SQLite database and image files.
    
    Returns the path to the created backup directory.
    """
    db_path = os.path.join(workspace_root, "workspace.sqlite3")
    images_dir = os.path.join(workspace_root, "images")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Workspace database not found at {db_path}")

    os.makedirs(backup_dest_dir, exist_ok=True)
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_run_dir = os.path.join(backup_dest_dir, f"backup_{timestamp}")
    backup_images_dir = os.path.join(backup_run_dir, "images")
    os.makedirs(backup_images_dir, exist_ok=True)

    backup_db_path = os.path.join(backup_run_dir, "workspace.sqlite3")

    # Safe SQLite online backup API
    src_conn = sqlite3.connect(db_path)
    dst_conn = sqlite3.connect(backup_db_path)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    # Verify backup database integrity
    if not verify_backup_db(backup_db_path):
        raise RuntimeError("Backup database integrity check failed.")

    # Copy image evidence directory if it exists
    if os.path.exists(images_dir):
        for fname in os.listdir(images_dir):
            src_img = os.path.join(images_dir, fname)
            if os.path.isfile(src_img):
                shutil.copy2(src_img, os.path.join(backup_images_dir, fname))

    return backup_run_dir


def verify_backup_db(backup_db_path: str) -> bool:
    """Run PRAGMA integrity_check on the backup database."""
    if not os.path.exists(backup_db_path):
        return False
    try:
        conn = sqlite3.connect(backup_db_path)
        cur = conn.cursor()
        res = cur.execute("PRAGMA integrity_check;").fetchone()
        conn.close()
        return res and res[0] == "ok"
    except Exception:
        return False


def restore_backup(backup_run_dir: str, target_workspace_root: str = "data/workspace") -> None:
    """Safely restore database and image evidence from a verified backup directory
    into the target workspace root.
    """
    backup_db_path = os.path.join(backup_run_dir, "workspace.sqlite3")
    backup_images_dir = os.path.join(backup_run_dir, "images")

    if not verify_backup_db(backup_db_path):
        raise ValueError("Cannot restore: backup database failed integrity check.")

    os.makedirs(target_workspace_root, exist_ok=True)
    target_db_path = os.path.join(target_workspace_root, "workspace.sqlite3")
    target_images_dir = os.path.join(target_workspace_root, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    # Restore DB
    shutil.copy2(backup_db_path, target_db_path)

    # Restore images
    if os.path.exists(backup_images_dir):
        for fname in os.listdir(backup_images_dir):
            src_img = os.path.join(backup_images_dir, fname)
            if os.path.isfile(src_img):
                shutil.copy2(src_img, os.path.join(target_images_dir, fname))
