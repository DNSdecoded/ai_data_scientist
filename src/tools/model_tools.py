import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.config import OUTPUTS_DIR
from src.exceptions import ModelTrainingError
from src.logger import log


class ModelTrainerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    target_column: str = Field(description="Name of the target column")
    task_type: str = Field(default="classification", description="Task type: classification or regression")


class ModelTrainer(BaseTool):
    name: str = "model_trainer"
    description: str = "Train multiple ML models and compare performance. Returns metrics for each model."
    args_schema: type[BaseModel] = ModelTrainerInput

    def _run(self, file_path: str, target_column: str, task_type: str = "classification") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ModelTrainingError(f"Failed to read file: {e}")

        if target_column not in df.columns:
            return f"Target column '{target_column}' not found."

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
        if not feature_cols:
            return "No numeric feature columns found for training."

        X = df[feature_cols].fillna(0)
        y = df[target_column]

        le = LabelEncoder()
        if y.dtype == "object":
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
        }

        results = []
        for name, model in models.items():
            start = time.time()
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                metrics = {
                    "model_name": name,
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "training_time_seconds": round(time.time() - start, 2),
                }

                if y_proba is not None and len(np.unique(y)) == 2:
                    metrics["auc"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)

                cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
                metrics["cv_score_mean"] = round(float(cv_scores.mean()), 4)
                metrics["cv_score_std"] = round(float(cv_scores.std()), 4)

                results.append(metrics)
                log.info("model_trained", model=name, f1=metrics["f1"], accuracy=metrics["accuracy"])
            except Exception as e:
                log.warning("model_training_failed", model=name, error=str(e))
                results.append({"model_name": name, "error": str(e)})

        best = max([r for r in results if "error" not in r], key=lambda x: x.get("f1", 0), default=None)

        output_path = OUTPUTS_DIR / "model_results.json"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"results": results, "best_model": best}, f, indent=2)

        return json.dumps({"results": results, "best_model": best}, indent=2)


class ModelEvaluatorInput(BaseModel):
    file_path: str = Field(description="Path to model results JSON")
    metric: str = Field(default="f1", description="Metric to rank by: f1, accuracy, precision, recall, auc")


class ModelEvaluator(BaseTool):
    name: str = "model_evaluator"
    description: str = "Evaluate and rank trained models from a results JSON file."
    args_schema: type[BaseModel] = ModelEvaluatorInput

    def _run(self, file_path: str, metric: str = "f1") -> str:
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            return f"Failed to read results file: {e}"

        results = data.get("results", [])
        valid = [r for r in results if "error" not in r and metric in r]
        ranked = sorted(valid, key=lambda x: x[metric], reverse=True)

        summary = []
        for i, r in enumerate(ranked, 1):
            summary.append(f"{i}. {r['model_name']}: {metric}={r[metric]} (acc={r.get('accuracy', 'N/A')})")

        return "\n".join(summary) if summary else "No valid model results found."
