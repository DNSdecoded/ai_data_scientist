import json

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

from src.exceptions import StatisticalAnalysisError
from src.logger import log


class DescriptiveStatsInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")


class DescriptiveStats(BaseTool):
    name: str = "descriptive_stats"
    description: str = "Compute descriptive statistics for all columns (mean, median, std, skew, kurtosis)."
    args_schema: type[BaseModel] = DescriptiveStatsInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise StatisticalAnalysisError(f"Failed to read file: {e}")

        stats = {}
        for col in df.columns:
            col_stats = {"dtype": str(df[col].dtype), "count": int(df[col].count())}
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                if len(s) > 0:
                    col_stats.update({
                        "mean": round(float(s.mean()), 4),
                        "median": round(float(s.median()), 4),
                        "std": round(float(s.std()), 4),
                        "skewness": round(float(s.skew()), 4),
                        "kurtosis": round(float(s.kurtosis()), 4),
                        "min": round(float(s.min()), 4),
                        "max": round(float(s.max()), 4),
                        "q25": round(float(s.quantile(0.25)), 4),
                        "q75": round(float(s.quantile(0.75)), 4),
                    })
            else:
                col_stats["unique"] = int(df[col].nunique())
                top = df[col].value_counts().head(3)
                col_stats["top_values"] = {str(k): int(v) for k, v in top.items()}
            stats[col] = col_stats

        log.info("descriptive_stats_computed", columns=len(stats))
        return json.dumps(stats)


class CorrelationAnalyzerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    method: str = Field(default="pearson", description="Correlation method: pearson, spearman, or kendall")


class CorrelationAnalyzer(BaseTool):
    name: str = "correlation_analyzer"
    description: str = "Compute correlation matrix between numeric columns."
    args_schema: type[BaseModel] = CorrelationAnalyzerInput

    def _run(self, file_path: str, method: str = "pearson") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise StatisticalAnalysisError(f"Failed to read file: {e}")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation analysis."

        corr = numeric_df.corr(method=method)
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong_pairs.append({
                        "col_1": corr.columns[i],
                        "col_2": corr.columns[j],
                        "correlation": round(float(val), 4),
                    })

        strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        log.info("correlation_analyzed", method=method, strong_pairs=len(strong_pairs))
        return json.dumps({"method": method, "strong_pairs": strong_pairs[:20], "matrix_shape": list(corr.shape)})


class HypothesisTesterInput(BaseModel):
    file_path: str = Field(description="Path to the CSV dataset")
    column: str = Field(description="Column to test")
    group_column: str = Field(default="", description="Grouping column for comparison tests")
    test_type: str = Field(default="auto", description="Test type: auto, ttest, anova, chi2, mannwhitney")


class HypothesisTester(BaseTool):
    name: str = "hypothesis_tester"
    description: str = "Run statistical hypothesis tests (t-test, ANOVA, chi-square, Mann-Whitney)."
    args_schema: type[BaseModel] = HypothesisTesterInput

    def _run(self, file_path: str, column: str, group_column: str = "", test_type: str = "auto") -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise StatisticalAnalysisError(f"Failed to read file: {e}")

        if column not in df.columns:
            return f"Column '{column}' not found."

        results = {}

        if group_column and group_column in df.columns:
            groups = df[group_column].unique()
            if len(groups) < 2:
                return "Need at least 2 groups for comparison."

            if test_type in ("auto", "ttest") and len(groups) == 2:
                g1 = df[df[group_column] == groups[0]][column].dropna()
                g2 = df[df[group_column] == groups[1]][column].dropna()
                stat, pval = scipy_stats.ttest_ind(g1, g2)
                results = {"test": "t-test", "statistic": round(float(stat), 4), "p_value": round(float(pval), 4),
                           "significant": bool(pval < 0.05)}
            elif test_type in ("auto", "anova"):
                group_data = [df[df[group_column] == g][column].dropna() for g in groups]
                stat, pval = scipy_stats.f_oneway(*group_data)
                results = {"test": "ANOVA", "statistic": round(float(stat), 4), "p_value": round(float(pval), 4),
                           "significant": bool(pval < 0.05)}
            elif test_type in ("auto", "mannwhitney"):
                g1 = df[df[group_column] == groups[0]][column].dropna()
                g2 = df[df[group_column] == groups[1]][column].dropna()
                stat, pval = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                results = {"test": "Mann-Whitney U", "statistic": round(float(stat), 4),
                           "p_value": round(float(pval), 4), "significant": bool(pval < 0.05)}
        else:
            if pd.api.types.is_numeric_dtype(df[column]):
                series = df[column].dropna()
                stat, pval = scipy_stats.shapiro(series.sample(min(5000, len(series))))
                results = {"test": "Shapiro-Wilk normality", "statistic": round(float(stat), 4),
                           "p_value": round(float(pval), 4), "normal": bool(pval > 0.05)}

        log.info("hypothesis_tested", test=results.get("test", "unknown"), p_value=results.get("p_value"))
        return json.dumps(results)
