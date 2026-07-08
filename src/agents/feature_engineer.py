from crewai import Agent

from src.config import get_settings
from src.tools.feature_tools import FeatureEncoder, FeatureCreator, FeatureSelector


def create_feature_engineer() -> Agent:
    settings = get_settings()
    return Agent(
        role="Feature Engineering Expert",
        goal="Transform raw data into meaningful features through encoding, creation, and selection to maximize model performance.",
        backstory=(
            "You are a creative feature engineer who can extract signal from noise. "
            "You know when to use one-hot vs label encoding, how to create meaningful interactions, "
            "and which features actually matter. Your engineered features consistently improve model accuracy."
        ),
        tools=[FeatureEncoder(), FeatureCreator(), FeatureSelector()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
