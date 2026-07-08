class DataScientistError(Exception):
    """Base exception for the AI Data Scientist platform."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataLoadError(DataScientistError):
    """Raised when dataset loading fails."""


class DataValidationError(DataScientistError):
    """Raised when dataset validation fails (size, columns, format)."""


class CleaningError(DataScientistError):
    """Raised when data cleaning fails."""


class FeatureEngineeringError(DataScientistError):
    """Raised when feature engineering fails."""


class ModelTrainingError(DataScientistError):
    """Raised when model training fails."""


class StatisticalAnalysisError(DataScientistError):
    """Raised when statistical analysis fails."""


class VisualizationError(DataScientistError):
    """Raised when chart generation fails."""


class ReportGenerationError(DataScientistError):
    """Raised when report generation fails."""


class LLMProviderError(DataScientistError):
    """Raised when LLM API calls fail after retries."""


class CrewExecutionError(DataScientistError):
    """Raised when the CrewAI crew execution fails."""
