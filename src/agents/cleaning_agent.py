from crewai import Agent

from src.config import get_settings
from src.tools.cleaning_tools import (
    MissingValueAnalyzer,
    MissingValueImputer,
    OutlierDetector,
    DataNormalizer,
)


def create_cleaning_agent() -> Agent:
    settings = get_settings()
    return Agent(
        role="Data Cleaning Specialist",
        goal="Clean datasets by handling missing values, detecting outliers, normalizing data, and ensuring data quality for reliable analysis.",
        backstory=(
            "You are a meticulous data cleaning expert who has cleaned thousands of datasets. "
            "You understand the nuances of missing data patterns (MCAR, MAR, MNAR), know when to impute "
            "vs drop, and can handle outliers without losing valuable information. Your cleaned datasets "
            "are always analysis-ready."
        ),
        tools=[MissingValueAnalyzer(), MissingValueImputer(), OutlierDetector(), DataNormalizer()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
