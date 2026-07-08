import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import EXPERIMENTS_DIR
from src.logger import log
from src.models.schemas import ExperimentResult, ModelMetrics


class ExperimentStore:
    def __init__(self):
        self.db_path = EXPERIMENTS_DIR / "experiments.db"
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_hash TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    timestamp TEXT,
                    models_json TEXT,
                    best_model TEXT,
                    best_f1 REAL,
                    feature_count INTEGER,
                    rows_used INTEGER,
                    report_path TEXT,
                    status TEXT DEFAULT 'completed'
                )
            """)
            conn.commit()

    def log_experiment(self, result: ExperimentResult) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO experiments
                   (dataset_hash, run_name, timestamp, models_json, best_model, best_f1,
                    feature_count, rows_used, report_path, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.dataset_hash,
                    result.run_name,
                    result.timestamp.isoformat(),
                    json.dumps([m.model_dump() for m in result.models]),
                    result.best_model,
                    result.best_f1,
                    result.feature_count,
                    result.rows_used,
                    result.report_path,
                    result.status,
                ),
            )
            conn.commit()
            exp_id = cursor.lastrowid

        log.info("experiment_logged", experiment_id=exp_id, best_model=result.best_model, best_f1=result.best_f1)
        return exp_id

    def get_experiment(self, experiment_id: int) -> ExperimentResult | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_result(dict(row))

    def list_experiments(self, limit: int = 50) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_result(dict(r)) for r in rows]

    def compare_experiments(self, ids: list[int]) -> list[dict]:
        results = []
        for eid in ids:
            exp = self.get_experiment(eid)
            if exp:
                results.append({
                    "id": exp.id,
                    "run_name": exp.run_name,
                    "best_model": exp.best_model,
                    "best_f1": exp.best_f1,
                    "feature_count": exp.feature_count,
                    "timestamp": exp.timestamp.isoformat(),
                })
        return results

    @staticmethod
    def _row_to_result(row: dict) -> ExperimentResult:
        models_data = json.loads(row.get("models_json") or "[]")
        models = [ModelMetrics(**m) for m in models_data]
        return ExperimentResult(
            id=row["id"],
            dataset_hash=row["dataset_hash"],
            run_name=row["run_name"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.utcnow(),
            models=models,
            best_model=row.get("best_model", ""),
            best_f1=row.get("best_f1", 0.0),
            feature_count=row.get("feature_count", 0),
            rows_used=row.get("rows_used", 0),
            report_path=row.get("report_path", ""),
            status=row.get("status", "completed"),
        )
