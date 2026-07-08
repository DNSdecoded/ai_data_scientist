import json
from pathlib import Path

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import PROCESSED_DIR
from src.exceptions import CleaningError
from src.logger import log


class MissingValueAnalyzerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")


class MissingValueAnalyzer(BaseTool):
    name: str = "missing_value_analyzer"
    description: str = "Analyze missing value patterns in a dataset. Returns missing counts, percentages, and pattern classification."
    args_schema: type[BaseModel] = MissingValueAnalyzerInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise CleaningError(f"Failed to read file: {e}")

        missing_info = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                missing_info[col] = {
                    "null_count": null_count,
                    "null_pct": round(null_count / len(df) * 100, 2),
                    "dtype": str(df[col].dtype),
                }

        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        pattern = "MCAR" if total_missing == 0 else self._classify_pattern(df)

        result = {
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "total_missing_cells": int(total_missing),
            "overall_missing_pct": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
            "pattern": pattern,
            "columns_with_missing": missing_info,
            "recommendations": self._generate_recommendations(missing_info),
        }

        log.info("missing_values_analyzed", file_path=file_path, columns_with_missing=len(missing_info))
        return json.dumps(result)

    @staticmethod
    def _classify_pattern(df: pd.DataFrame) -> str:
        missing_cols = [c for c in df.columns if df[c].isnull().any()]
        if len(missing_cols) < 2:
            return "MCAR"
        subset = df[missing_cols].isnull().astype(int)
        corr = subset.corr()
        off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
        if any(abs(v) > 0.3 for v in off_diag):
            return "MAR"
        return "MCAR"

    @staticmethod
    def _generate_recommendations(missing_info: dict) -> list[str]:
        recs = []
        for col, info in missing_info.items():
            pct = info["null_pct"]
            if pct > 50:
                recs.append(f"Column '{col}' has {pct}% missing - consider dropping")
            elif pct > 20:
                recs.append(f"Column '{col}' has {pct}% missing - use advanced imputation (KNN)")
            elif pct > 5:
                recs.append(f"Column '{col}' has {pct}% missing - use median/mode imputation")
            else:
                recs.append(f"Column '{col}' has {pct}% missing - use mean/median imputation")
        return recs


class MissingValueImputerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    strategy: str = Field(default="auto", description="Imputation strategy: auto, mean, median, mode, drop")


class MissingValueImputer(BaseTool):
    name: str = "missing_value_imputer"
    description: str = "Impute missing values in a dataset. Supports auto, mean, median, mode, and drop strategies."
    args_schema: type[BaseModel] = MissingValueImputerInput

    def _run(self, file_path: str, strategy: str = "auto") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise CleaningError(f"Failed to read file: {e}")

        before_missing = int(df.isnull().sum().sum())

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "auto":
            for col in df.columns:
                if df[col].isnull().sum() == 0:
                    continue
                if df[col].dtype in ("float64", "int64"):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        elif strategy == "mean":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            for col in df.columns:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])

        after_missing = int(df.isnull().sum().sum())
        output_path = PROCESSED_DIR / f"cleaned_{Path(file_path).name}"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info("missing_values_imputed", strategy=strategy, before=before_missing, after=after_missing)
        return f"Imputed {before_missing - after_missing} missing values using '{strategy}' strategy. Saved to {output_path}. Remaining missing: {after_missing}"


class OutlierDetectorInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    method: str = Field(default="iqr", description="Detection method: iqr or zscore")
    threshold: float = Field(default=1.5, description="Threshold for IQR (default 1.5) or Z-score (default 3.0)")


class OutlierDetector(BaseTool):
    name: str = "outlier_detector"
    description: str = "Detect outliers in numeric columns using IQR or Z-score method."
    args_schema: type[BaseModel] = OutlierDetectorInput

    def _run(self, file_path: str, method: str = "iqr", threshold: float = 1.5) -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise CleaningError(f"Failed to read file: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outliers = series[(series < lower) | (series > upper)]
            else:
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue
                z_scores = np.abs((series - mean) / std)
                outliers = series[z_scores > threshold]

            if len(outliers) > 0:
                outlier_report[col] = {
                    "count": len(outliers),
                    "pct": round(len(outliers) / len(series) * 100, 2),
                    "min_outlier": float(outliers.min()),
                    "max_outlier": float(outliers.max()),
                }

        log.info("outliers_detected", method=method, columns_with_outliers=len(outlier_report))
        return json.dumps(outlier_report)


class DataNormalizerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    method: str = Field(default="standard", description="Normalization method: standard, minmax, or robust")


class DataNormalizer(BaseTool):
    name: str = "data_normalizer"
    description: str = "Normalize/standardize numeric columns in a dataset."
    args_schema: type[BaseModel] = DataNormalizerInput

    def _run(self, file_path: str, method: str = "standard") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise CleaningError(f"Failed to read file: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return "No numeric columns to normalize."

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0 or series.std() == 0:
                continue

            if method == "standard":
                df[col] = (df[col] - series.mean()) / series.std()
            elif method == "minmax":
                min_val = series.min()
                max_val = series.max()
                if max_val - min_val != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == "robust":
                median = series.median()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    df[col] = (df[col] - median) / iqr

        output_path = PROCESSED_DIR / f"normalized_{Path(file_path).name}"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info("data_normalized", method=method, columns=len(numeric_cols))
        return f"Normalized {len(numeric_cols)} numeric columns using '{method}' method. Saved to {output_path}"
