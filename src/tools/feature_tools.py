from pathlib import Path

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import PROCESSED_DIR
from src.exceptions import FeatureEngineeringError
from src.logger import log


class FeatureEncoderInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    method: str = Field(default="auto", description="Encoding method: auto, onehot, label, or target")
    target_column: str = Field(default="", description="Target column for target encoding (if method=target)")


class FeatureEncoder(BaseTool):
    name: str = "feature_encoder"
    description: str = "Encode categorical features. Supports one-hot, label, and target encoding."
    args_schema: type[BaseModel] = FeatureEncoderInput

    def _run(self, file_path: str, method: str = "auto", target_column: str = "") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to read file: {e}")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return "No categorical columns to encode."

        encoded_count = 0
        for col in cat_cols:
            n_unique = df[col].nunique()

            if method == "auto":
                if n_unique <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                    encoded_count += 1
                else:
                    df[col] = df[col].astype("category").cat.codes
                    encoded_count += 1
            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                encoded_count += 1
            elif method == "label":
                df[col] = df[col].astype("category").cat.codes
                encoded_count += 1
            elif method == "target" and target_column and target_column in df.columns:
                means = df.groupby(col)[target_column].mean()
                df[col] = df[col].map(means)
                encoded_count += 1

        output_path = PROCESSED_DIR / f"encoded_{Path(file_path).name}"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info("features_encoded", method=method, columns_encoded=encoded_count)
        return f"Encoded {encoded_count} categorical columns using '{method}'. Saved to {output_path}"


class FeatureCreatorInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    operations: str = Field(default="interactions", description="Feature creation operations: interactions, polynomial, log")


class FeatureCreator(BaseTool):
    name: str = "feature_creator"
    description: str = "Create new features from existing numeric columns (interactions, polynomial, log transforms)."
    args_schema: type[BaseModel] = FeatureCreatorInput

    def _run(self, file_path: str, operations: str = "interactions") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to read file: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        created = []

        if "interactions" in operations and len(numeric_cols) >= 2:
            for i in range(min(len(numeric_cols), 5)):
                for j in range(i + 1, min(len(numeric_cols), 5)):
                    col_a, col_b = numeric_cols[i], numeric_cols[j]
                    new_col = f"{col_a}_x_{col_b}"
                    df[new_col] = df[col_a] * df[col_b]
                    created.append(new_col)

        if "log" in operations:
            for col in numeric_cols[:5]:
                if df[col].min() >= 0:
                    new_col = f"{col}_log"
                    df[new_col] = np.log1p(df[col])
                    created.append(new_col)

        if "polynomial" in operations and len(numeric_cols) >= 1:
            for col in numeric_cols[:3]:
                new_col = f"{col}_sq"
                df[new_col] = df[col] ** 2
                created.append(new_col)

        output_path = PROCESSED_DIR / f"features_{Path(file_path).name}"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info("features_created", operations=operations, new_features=len(created))
        return f"Created {len(created)} new features. Columns: {created[:10]}. Saved to {output_path}"


class FeatureSelectorInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    target_column: str = Field(description="Target column name")
    method: str = Field(default="correlation", description="Selection method: correlation, variance, or mutual_info")
    top_k: int = Field(default=20, description="Number of top features to select")


class FeatureSelector(BaseTool):
    name: str = "feature_selector"
    description: str = "Select top features based on correlation, variance threshold, or mutual information."
    args_schema: type[BaseModel] = FeatureSelectorInput

    def _run(self, file_path: str, target_column: str, method: str = "correlation", top_k: int = 20) -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to read file: {e}")

        if target_column not in df.columns:
            return f"Target column '{target_column}' not found."

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
        if not feature_cols:
            return "No numeric feature columns found."

        X = df[feature_cols].fillna(0)
        y = df[target_column].fillna(0)

        if method == "correlation":
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected = correlations.head(top_k).index.tolist()
        elif method == "variance":
            variances = X.var().sort_values(ascending=False)
            selected = variances.head(top_k).index.tolist()
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y)
            mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
            selected = mi_series.head(top_k).index.tolist()
        else:
            selected = feature_cols[:top_k]

        log.info("features_selected", method=method, selected_count=len(selected))
        return f"Selected {len(selected)} features using '{method}': {selected}"
