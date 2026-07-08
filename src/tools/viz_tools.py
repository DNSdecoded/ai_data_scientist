import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import OUTPUTS_DIR
from src.exceptions import VisualizationError
from src.logger import log


class ChartGeneratorInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    chart_types: str = Field(default="distributions,heatmap,boxplot", description="Comma-separated chart types")


class ChartGenerator(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate visualization charts: distributions, correlation heatmap, box plots, scatter plots."
    args_schema: type[BaseModel] = ChartGeneratorInput

    def _run(self, file_path: str, chart_types: str = "distributions,heatmap,boxplot") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise VisualizationError(f"Failed to read file: {e}")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        charts = []
        types = [t.strip() for t in chart_types.split(",")]

        sns.set_theme(style="whitegrid", palette="husl")

        if "distributions" in types:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        df[col].hist(ax=axes[i], bins=30, edgecolor="black", alpha=0.7)
                        axes[i].set_title(f"Distribution: {col}")
                for j in range(len(numeric_cols), len(axes)):
                    axes[j].set_visible(False)
                plt.tight_layout()
                path = OUTPUTS_DIR / "distributions.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                charts.append(str(path))

        if "heatmap" in types:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(12, 10))
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                            center=0, square=True, linewidths=0.5, ax=ax)
                ax.set_title("Correlation Heatmap")
                plt.tight_layout()
                path = OUTPUTS_DIR / "heatmap.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                charts.append(str(path))

        if "boxplot" in types:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                df[numeric_cols].boxplot(ax=ax)
                ax.set_title("Box Plots")
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = OUTPUTS_DIR / "boxplots.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                charts.append(str(path))

        log.info("charts_generated", count=len(charts))
        return json.dumps({"charts": charts, "count": len(charts)})


class ModelVizInput(BaseModel):
    results_file: str = Field(description="Path to model results JSON file")


class ModelViz(BaseTool):
    name: str = "model_viz"
    description: str = "Generate model comparison charts from experiment results."
    args_schema: type[BaseModel] = ModelVizInput

    def _run(self, results_file: str) -> str:
        try:
            with open(results_file) as f:
                data = json.load(f)
        except Exception as e:
            return f"Failed to read results: {e}"

        results = data.get("results", [])
        valid = [r for r in results if "error" not in r]
        if not valid:
            return "No valid model results to visualize."

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        charts = []

        names = [r["model_name"] for r in valid]
        metrics = ["accuracy", "precision", "recall", "f1"]
        available_metrics = [m for m in metrics if any(m in r for r in valid)]

        if available_metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(names))
            width = 0.2
            for i, metric in enumerate(available_metrics):
                values = [r.get(metric, 0) for r in valid]
                ax.bar(x + i * width, values, width, label=metric.capitalize())
            ax.set_xlabel("Models")
            ax.set_ylabel("Score")
            ax.set_title("Model Comparison")
            ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
            ax.set_xticklabels(names)
            ax.legend()
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            path = OUTPUTS_DIR / "model_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            charts.append(str(path))

        if any("cv_score_mean" in r for r in valid):
            fig, ax = plt.subplots(figsize=(10, 5))
            cv_names = [r["model_name"] for r in valid if "cv_score_mean" in r]
            cv_means = [r["cv_score_mean"] for r in valid if "cv_score_mean" in r]
            cv_stds = [r.get("cv_score_std", 0) for r in valid if "cv_score_mean" in r]
            ax.barh(cv_names, cv_means, xerr=cv_stds, capsize=5, color=sns.color_palette("husl", len(cv_names)))
            ax.set_xlabel("CV Accuracy")
            ax.set_title("Cross-Validation Scores")
            plt.tight_layout()
            path = OUTPUTS_DIR / "cv_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            charts.append(str(path))

        log.info("model_charts_generated", count=len(charts))
        return json.dumps({"charts": charts, "count": len(charts)})
