"""SQLite-backed inspection workspace store.

Image bytes are stored under data/workspace/images/<sha256>.<ext>.
Filenames from user input are NEVER used as paths. Original bytes are preserved
verbatim and identified only by their SHA-256.
"""

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from workspace.models import (
    Asset, Inspection, Finding, MaintenanceRecord,
    HUMAN_REVIEW_STATUSES,
)

DEFAULT_ROOT = "data/workspace"
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB per image


class WorkspaceStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.db_path = os.path.join(root, "workspace.sqlite3")
        os.makedirs(self.images_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        cur = self._conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            pipe_material TEXT NOT NULL DEFAULT 'unknown',
            pipe_type TEXT NOT NULL DEFAULT 'unknown',
            criticality TEXT NOT NULL DEFAULT 'normal',
            notes TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            inspection_date TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            image_ext TEXT NOT NULL,
            annotated_sha256 TEXT,
            detector_results_json TEXT NOT NULL DEFAULT '{}',
            authoritative_report_json TEXT NOT NULL DEFAULT '{}',
            inspector_notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
        );
        CREATE TABLE IF NOT EXISTS findings (
            finding_id TEXT PRIMARY KEY,
            inspection_id TEXT NOT NULL,
            detector_id TEXT NOT NULL,
            defect_category TEXT NOT NULL,
            bbox_json TEXT,
            confidence REAL,
            evidence_description TEXT NOT NULL DEFAULT '',
            limitations_json TEXT NOT NULL DEFAULT '[]',
            human_review_status TEXT NOT NULL DEFAULT 'unreviewed',
            review_notes TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT,
            FOREIGN KEY(inspection_id) REFERENCES inspections(inspection_id)
        );
        CREATE TABLE IF NOT EXISTS maintenance_records (
            record_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            inspection_id TEXT,
            work_performed TEXT NOT NULL,
            performed_by TEXT NOT NULL,
            performed_at TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'recorded_work'
        );
        INSERT OR IGNORE INTO schema_version(version) VALUES (1);
        """)
        self._conn.commit()

    # ---- image storage ----
    def _sniff_ext(self, data: bytes) -> Optional[str]:
        for magic, ext in IMAGE_MAGIC.items():
            if data.startswith(magic):
                return ext
        # webp: RIFF....WEBP
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        return None

    def store_image(self, data: bytes) -> Tuple[str, str]:
        """Validate, deduplicate by SHA-256, and store an image. Returns (sha256, ext)."""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("image data must be bytes")
        if len(data) == 0:
            raise ValueError("image is empty")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
        ext = self._sniff_ext(bytes(data))
        if ext is None:
            raise ValueError("unsupported or unrecognised image format (only jpg/png/webp accepted)")
        digest = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.images_dir, f"{digest}.{ext}")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return digest, ext

    def image_path(self, sha256: str, ext: str) -> str:
        # sha256 validated to be hex only, ext validated from sniff
        if not all(c in "0123456789abcdef" for c in sha256) or len(sha256) != 64:
            raise ValueError("invalid image identifier")
        if ext not in ("jpg", "png", "webp"):
            raise ValueError("invalid image extension")
        return os.path.join(self.images_dir, f"{sha256}.{ext}")

    # ---- assets ----
    def upsert_asset(self, asset: Asset) -> None:
        if not asset.asset_id or len(asset.asset_id) > 128:
            raise ValueError("asset_id must be 1..128 chars")
        if asset.criticality not in ("low", "normal", "high", "critical"):
            raise ValueError("invalid criticality")
        self._conn.execute(
            "INSERT INTO assets(asset_id, location, pipe_material, pipe_type, criticality, notes) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(asset_id) DO UPDATE SET location=excluded.location, "
            "pipe_material=excluded.pipe_material, pipe_type=excluded.pipe_type, "
            "criticality=excluded.criticality, notes=excluded.notes",
            (asset.asset_id, asset.location, asset.pipe_material,
             asset.pipe_type, asset.criticality, asset.notes),
        )
        self._conn.commit()

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        row = self._conn.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if not row:
            return None
        return Asset(**{k: row[k] for k in row.keys()})

    def list_assets(self) -> List[Asset]:
        rows = self._conn.execute("SELECT * FROM assets ORDER BY asset_id").fetchall()
        return [Asset(**{k: r[k] for k in r.keys()}) for r in rows]

    # ---- inspections ----
    def create_inspection(self, insp: Inspection) -> None:
        self._conn.execute(
            "INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (insp.inspection_id, insp.asset_id, insp.inspection_date,
             insp.image_sha256, insp.image_ext, insp.annotated_sha256,
             _dumps(insp.detector_results), _dumps(insp.authoritative_report),
             insp.inspector_notes, insp.created_at or _now_iso()),
        )
        self._conn.commit()

    def get_inspection(self, inspection_id: str) -> Optional[Inspection]:
        row = self._conn.execute(
            "SELECT * FROM inspections WHERE inspection_id=?", (inspection_id,)
        ).fetchone()
        if not row:
            return None
        return Inspection(
            inspection_id=row["inspection_id"],
            asset_id=row["asset_id"],
            inspection_date=row["inspection_date"],
            image_sha256=row["image_sha256"],
            image_ext=row["image_ext"],
            annotated_sha256=row["annotated_sha256"],
            detector_results=_loads(row["detector_results_json"]),
            authoritative_report=_loads(row["authoritative_report_json"]),
            inspector_notes=row["inspector_notes"],
            created_at=row["created_at"],
        )

    def list_inspections_for_asset(self, asset_id: str) -> List[Inspection]:
        rows = self._conn.execute(
            "SELECT * FROM inspections WHERE asset_id=? ORDER BY inspection_date DESC",
            (asset_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(Inspection(
                inspection_id=r["inspection_id"], asset_id=r["asset_id"],
                inspection_date=r["inspection_date"], image_sha256=r["image_sha256"],
                image_ext=r["image_ext"], annotated_sha256=r["annotated_sha256"],
                detector_results=_loads(r["detector_results_json"]),
                authoritative_report=_loads(r["authoritative_report_json"]),
                inspector_notes=r["inspector_notes"], created_at=r["created_at"],
            ))
        return out

    # ---- findings ----
    def add_findings(self, findings: List[Finding]) -> None:
        for f in findings:
            if f.human_review_status not in HUMAN_REVIEW_STATUSES:
                raise ValueError(f"invalid review status: {f.human_review_status}")
            self._conn.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f.finding_id, f.inspection_id, f.detector_id, f.defect_category,
                 _dumps(f.bbox), f.confidence, f.evidence_description,
                 _dumps(f.limitations), f.human_review_status,
                 f.review_notes, f.reviewed_at),
            )
        self._conn.commit()

    def list_findings(self, inspection_id: str) -> List[Finding]:
        rows = self._conn.execute(
            "SELECT * FROM findings WHERE inspection_id=? ORDER BY defect_category, finding_id",
            (inspection_id,),
        ).fetchall()
        return [_finding_from_row(r) for r in rows]

    def review_finding(self, finding_id: str, status: str, notes: str = "") -> None:
        if status not in HUMAN_REVIEW_STATUSES:
            raise ValueError(f"invalid review status: {status}")
        cur = self._conn.execute(
            "UPDATE findings SET human_review_status=?, review_notes=?, reviewed_at=? WHERE finding_id=?",
            (status, notes, _now_iso(), finding_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"finding not found: {finding_id}")
        self._conn.commit()

    # ---- maintenance records ----
    def add_maintenance_record(self, rec: MaintenanceRecord) -> None:
        if rec.kind not in ("recorded_work", "recommendation"):
            raise ValueError("invalid maintenance record kind")
        self._conn.execute(
            "INSERT INTO maintenance_records VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rec.record_id, rec.asset_id, rec.inspection_id,
             rec.work_performed, rec.performed_by, rec.performed_at, rec.kind),
        )
        self._conn.commit()

    def list_maintenance_records(self, asset_id: str) -> List[MaintenanceRecord]:
        rows = self._conn.execute(
            "SELECT * FROM maintenance_records WHERE asset_id=? ORDER BY performed_at DESC",
            (asset_id,),
        ).fetchall()
        return [MaintenanceRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def close(self):
        self._conn.close()


def _dumps(v: Any) -> str:
    import json
    return json.dumps(v, sort_keys=True)


def _loads(s: str) -> Any:
    import json
    return json.loads(s) if s else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finding_from_row(r: sqlite3.Row) -> Finding:
    return Finding(
        finding_id=r["finding_id"], inspection_id=r["inspection_id"],
        detector_id=r["detector_id"], defect_category=r["defect_category"],
        bbox=_loads(r["bbox_json"]),
        confidence=r["confidence"],
        evidence_description=r["evidence_description"],
        limitations=_loads(r["limitations_json"]) or [],
        human_review_status=r["human_review_status"],
        review_notes=r["review_notes"],
        reviewed_at=r["reviewed_at"],
    )