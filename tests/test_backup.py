"""Unit tests for Phase 3 SQLite backup, integrity check, and restoration."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workspace.store import WorkspaceStore
from workspace.models import Asset
from workspace.backup import create_backup, verify_backup_db, restore_backup


def test_backup_and_restore_workflow(tmp_path):
    ws_root = str(tmp_path / "workspace")
    backup_root = str(tmp_path / "backups")
    restore_root = str(tmp_path / "restored")

    # 1. Initialize store and add data
    store = WorkspaceStore(root=ws_root)
    store.upsert_asset(Asset(asset_id="PIPE-BK-01", location="Backup Site", criticality="critical"))
    sha, ext = store.store_image(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    store.close()

    # 2. Create backup
    backup_dir = create_backup(workspace_root=ws_root, backup_dest_dir=backup_root)
    assert os.path.exists(backup_dir)
    assert os.path.exists(os.path.join(backup_dir, "workspace.sqlite3"))

    # 3. Verify backup db integrity
    assert verify_backup_db(os.path.join(backup_dir, "workspace.sqlite3")) is True

    # 4. Restore backup into isolated temporary location
    restore_backup(backup_dir, target_workspace_root=restore_root)
    assert os.path.exists(os.path.join(restore_root, "workspace.sqlite3"))

    # 5. Verify restored store data
    restored_store = WorkspaceStore(root=restore_root)
    try:
        asset = restored_store.get_asset("PIPE-BK-01")
        assert asset is not None
        assert asset.criticality == "critical"
    finally:
        restored_store.close()
