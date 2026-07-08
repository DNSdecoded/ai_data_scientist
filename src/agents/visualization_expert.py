from crewai import Agent

from src.config import get_settings
from src.tools.viz_tools import ChartGenerator, ModelViz


def create_visualization_expert() -> Agent:
    settings = get_settings()
    return Agent(
        role="Data Visualization Expert",
        goal="Create clear, insightful visualizations that communicate data patterns, distributions, and model performance effectively.",
        backstory=(
            "You are a data visualization specialist who transforms complex data into compelling visual stories. "
            "You know when to use which chart type, how to choose effective color palettes, and how to "
            "make charts that are both beautiful and informative. Your visualizations help stakeholders "
            "understand data at a glance."
        ),
        tools=[ChartGenerator(), ModelViz()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
