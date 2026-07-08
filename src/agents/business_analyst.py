from crewai import Agent

from src.config import get_settings
from src.tools.report_tools import ReportGenerator


def create_business_analyst() -> Agent:
    settings = get_settings()
    return Agent(
        role="Senior Business Analyst",
        goal="Synthesize all findings into a clear executive report with actionable insights and recommendations.",
        backstory=(
            "You are a business-savvy analyst who bridges the gap between technical data science and "
            "business decisions. You excel at translating complex analytical findings into clear, actionable "
            "recommendations. Your reports are always concise, well-structured, and focused on business impact."
        ),
        tools=[ReportGenerator()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
