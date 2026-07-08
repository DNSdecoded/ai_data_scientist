from datetime import UTC, datetime
from enum import Enum
from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    column_type: ColumnType
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    mean: float | None = None
    std: float | None = None
    min_val: str | None = None
    max_val: str | None = None
    top_values: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    file_path: str
    file_hash: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    total_missing_pct: float = 0.0
    memory_usage_mb: float = 0.0
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetVersion(BaseModel):
    version_id: str
    file_hash: str
    original_name: str
    row_count: int
    column_count: int
    version_schema: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float | None = None
    cv_score_mean: float = 0.0
    cv_score_std: float = 0.0
    training_time_seconds: float = 0.0


class ExperimentResult(BaseModel):
    id: int | None = None
    dataset_hash: str
    run_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    models: list[ModelMetrics] = Field(default_factory=list)
    best_model: str = ""
    best_f1: float = 0.0
    feature_count: int = 0
    rows_used: int = 0
    report_path: str = ""
    status: str = "completed"
