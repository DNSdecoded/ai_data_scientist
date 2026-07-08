import json

from src.tools.data_tools import DataLoaderTool, SchemaInspectorTool
from src.tools.cleaning_tools import MissingValueAnalyzer, MissingValueImputer
from src.tools.feature_tools import FeatureEncoder, FeatureCreator
from src.tools.model_tools import ModelTrainer
from src.tools.stats_tools import DescriptiveStats, CorrelationAnalyzer


class TestDataLoader:
    def test_load_csv(self, sample_csv):
        tool = DataLoaderTool()
        result = tool._run(sample_csv)
        data = json.loads(result)
        assert data["row_count"] == 8
        assert data["column_count"] == 5

    def test_load_nonexistent(self):
        tool = DataLoaderTool()
        try:
            tool._run("nonexistent.csv")
            assert False, "Should have raised DataLoadError"
        except Exception as e:
            assert "not found" in str(e).lower() or "File" in str(e)


class TestSchemaInspector:
    def test_inspect(self, sample_csv):
        tool = SchemaInspectorTool()
        result = tool._run(sample_csv)
        data = json.loads(result)
        assert len(data["columns"]) == 5
        assert data["row_count"] == 8


class TestMissingValueAnalyzer:
    def test_analyze(self, sample_csv):
        tool = MissingValueAnalyzer()
        result = tool._run(sample_csv)
        assert "total_missing_cells" in result or "null" in result.lower()


class TestMissingValueImputer:
    def test_impute_auto(self, sample_csv):
        tool = MissingValueImputer()
        result = tool._run(sample_csv, strategy="auto")
        assert "Imputed" in result or "imputed" in result.lower()


class TestFeatureEncoder:
    def test_encode_auto(self, sample_csv):
        tool = FeatureEncoder()
        result = tool._run(sample_csv, method="auto")
        assert "Encoded" in result or "encoded" in result.lower() or "No categorical" in result


class TestModelTrainer:
    def test_train(self, sample_csv):
        tool = ModelTrainer()
        result = tool._run(sample_csv, target_column="target")
        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) > 0


class TestDescriptiveStats:
    def test_describe(self, sample_csv):
        tool = DescriptiveStats()
        result = tool._run(sample_csv)
        assert "age" in result or "salary" in result


class TestCorrelationAnalyzer:
    def test_correlate(self, sample_csv):
        tool = CorrelationAnalyzer()
        result = tool._run(sample_csv, method="pearson")
        assert "correlation" in result.lower() or "pairs" in result.lower()
