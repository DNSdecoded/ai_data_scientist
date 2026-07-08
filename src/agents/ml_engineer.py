from crewai import Agent

from src.config import get_settings
from src.tools.model_tools import ModelTrainer, ModelEvaluator


def create_ml_engineer() -> Agent:
    settings = get_settings()
    return Agent(
        role="Senior ML Engineer",
        goal="Train, evaluate, and compare multiple machine learning models to find the best performer for the given task.",
        backstory=(
            "You are an experienced ML engineer who has built production models across industries. "
            "You know which algorithms work best for different data types and sizes, how to tune hyperparameters "
            "effectively, and how to evaluate models rigorously. You always consider both performance and "
            "interpretability."
        ),
        tools=[ModelTrainer(), ModelEvaluator()],
        llm=settings.resolved_model,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.agent_max_iter,
        max_rpm=settings.max_rpm,
    )
