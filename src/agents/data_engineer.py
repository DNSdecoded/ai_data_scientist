from crewai import Agent

from src.config import get_settings
from src.tools.data_tools import DataLoaderTool, SchemaInspectorTool


def create_data_engineer() -> Agent:
    settings = get_settings()
    return Agent(
        role="Senior Data Engineer",
        goal="Load datasets efficiently, detect schema, profile columns, and provide a comprehensive data overview for downstream analysis.",
        backstory=(
            "You are a seasoned data engineer with 10+ years of experience handling diverse data sources. "
            "You excel at quickly profiling datasets, identifying data types, detecting quality issues, "
            "and preparing data for analysis. You always provide clear, structured summaries of what you find."
        ),
        tools=[DataLoaderTool(), SchemaInspectorTool()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
