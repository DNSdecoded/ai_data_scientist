import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.config import VERSIONS_DIR, EXPERIMENTS_DIR
from src.logger import log
from src.models.schemas import DatasetVersion


class DatasetVersioner:
    def __init__(self):
        self.db_path = EXPERIMENTS_DIR / "versions.db"
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    row_count INTEGER,
                    column_count INTEGER,
                    version_schema TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def create_version(self, file_hash: str, original_name: str, file_path: str,
                       row_count: int, column_count: int, version_schema: str) -> DatasetVersion:
        version_id = f"v_{file_hash}_{int(datetime.now(UTC).timestamp())}"
        version = DatasetVersion(
            version_id=version_id,
            file_hash=file_hash,
            original_name=original_name,
            row_count=row_count,
            column_count=column_count,
            version_schema=version_schema,
        )

        dest_dir = VERSIONS_DIR / file_hash
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / original_name
        if not dest_path.exists():
            import shutil
            shutil.copy2(file_path, dest_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO dataset_versions
                   (version_id, file_hash, original_name, file_path, row_count, column_count, version_schema, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (version_id, file_hash, original_name, str(dest_path),
                 row_count, column_count, version_schema, version.created_at.isoformat()),
            )
            conn.commit()

        log.info("dataset_versioned", version_id=version_id, hash=file_hash)
        return version

    def get_version(self, version_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM dataset_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_versions(self, file_hash: str | None = None) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if file_hash:
                rows = conn.execute(
                    "SELECT * FROM dataset_versions WHERE file_hash = ? ORDER BY created_at DESC",
                    (file_hash,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM dataset_versions ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def diff_versions(self, v1_id: str, v2_id: str) -> dict:
        v1 = self.get_version(v1_id)
        v2 = self.get_version(v2_id)
        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        schema1 = json.loads(v1["version_schema"]) if v1["version_schema"] else {}
        schema2 = json.loads(v2["version_schema"]) if v2["version_schema"] else {}

        return {
            "version_1": v1_id,
            "version_2": v2_id,
            "row_count_change": v2["row_count"] - v1["row_count"],
            "column_count_change": v2["column_count"] - v1["column_count"],
            "schema_changed": schema1 != schema2,
        }
