import hashlib
import warnings
from pathlib import Path

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import UPLOADS_DIR, get_settings
from src.exceptions import DataLoadError, DataValidationError
from src.models.schemas import ColumnProfile, ColumnType, DatasetInfo
from src.logger import log


class DataLoaderInput(BaseModel):
    file_path: str = Field(description="Path to the dataset file (CSV, Excel, JSON, Parquet)")


class DataLoaderTool(BaseTool):
    name: str = "data_loader"
    description: str = "Load a dataset from a file path. Supports CSV, Excel, JSON, and Parquet formats."
    args_schema: type[BaseModel] = DataLoaderInput

    def _run(self, file_path: str) -> str:
        settings = get_settings()
        path = Path(file_path)

        if not path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise DataValidationError(
                f"File too large: {file_size_mb:.1f}MB exceeds limit of {settings.max_file_size_mb}MB"
            )

        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix in (".xls", ".xlsx"):
                df = pd.read_excel(path)
            elif suffix == ".json":
                df = pd.read_json(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise DataLoadError(f"Unsupported file format: {suffix}")
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to load file: {e}")

        if df.shape[1] > settings.max_columns:
            raise DataValidationError(
                f"Too many columns: {df.shape[1]} exceeds limit of {settings.max_columns}"
            )

        file_hash = self._hash_file(path)
        output_path = UPLOADS_DIR / f"{file_hash}_{path.name}"
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info("dataset_loaded", file_path=file_path, rows=len(df), columns=df.shape[1], hash=file_hash)

        info = DatasetInfo(
            file_path=str(output_path),
            file_hash=file_hash,
            row_count=len(df),
            column_count=df.shape[1],
            columns=[],
            total_missing_pct=float(df.isnull().mean().mean() * 100),
            memory_usage_mb=float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        )

        return info.model_dump_json(indent=2)

    @staticmethod
    def _hash_file(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]


class SchemaInspectorInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset to inspect")


class SchemaInspectorTool(BaseTool):
    name: str = "schema_inspector"
    description: str = "Inspect the schema of a dataset: column types, missing values, statistics, unique counts."
    args_schema: type[BaseModel] = SchemaInspectorInput

    def _run(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataLoadError(f"Failed to read file for inspection: {e}")

        profiles = []
        for col in df.columns:
            series = df[col]
            col_type = self._detect_type(series)

            profile = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                column_type=col_type,
                null_count=int(series.isnull().sum()),
                null_pct=round(float(series.isnull().mean() * 100), 2),
                unique_count=int(series.nunique()),
            )

            if col_type == ColumnType.NUMERIC:
                profile.mean = round(float(series.mean()), 4) if not series.empty else None
                profile.std = round(float(series.std()), 4) if not series.empty else None
                profile.min_val = str(series.min())
                profile.max_val = str(series.max())
            elif col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                top_vals = series.value_counts().head(5).index.tolist()
                profile.top_values = [str(v) for v in top_vals]

            profiles.append(profile)

        info = DatasetInfo(
            file_path=file_path,
            file_hash=DataLoaderTool._hash_file(path),
            row_count=len(df),
            column_count=df.shape[1],
            columns=profiles,
            total_missing_pct=round(float(df.isnull().mean().mean() * 100), 2),
            memory_usage_mb=round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
        )

        log.info("schema_inspected", file_path=file_path, columns=len(profiles))
        return info.model_dump_json(indent=2)

    @staticmethod
    def _detect_type(series: pd.Series) -> ColumnType:
        if series.dtype == bool:
            return ColumnType.BOOLEAN
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 2:
                return ColumnType.BOOLEAN
            return ColumnType.NUMERIC
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        sample = series.dropna().head(100)
        if len(sample) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                return ColumnType.DATETIME
        if series.nunique() < min(50, len(series) * 0.05):
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT
