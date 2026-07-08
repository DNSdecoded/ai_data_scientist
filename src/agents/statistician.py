from crewai import Agent

from src.config import get_settings
from src.tools.stats_tools import DescriptiveStats, CorrelationAnalyzer, HypothesisTester


def create_statistician() -> Agent:
    settings = get_settings()
    return Agent(
        role="Lead Statistician",
        goal="Perform rigorous statistical analysis to uncover patterns, test hypotheses, and validate findings in the data.",
        backstory=(
            "You are a PhD-trained statistician with expertise in applied statistics for data science. "
            "You know when to use parametric vs non-parametric tests, how to interpret p-values correctly, "
            "and how to communicate statistical findings in plain language. You always check assumptions "
            "before running tests."
        ),
        tools=[DescriptiveStats(), CorrelationAnalyzer(), HypothesisTester()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
