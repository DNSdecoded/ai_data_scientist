from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
VERSIONS_DIR = DATA_DIR / "versions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

LLM_MODELS = {
    "gemini": "gemini/gemini-3.1-flash-lite",
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-sonnet-4-20250514",
}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    llm_provider: str = Field(default="gemini", description="LLM provider: gemini | openai | anthropic")
    llm_model: str = Field(default="gemini/gemini-3.1-flash-lite", description="Full model identifier")

    max_rpm: int = Field(default=10, description="Max requests per minute for LLM calls (set to your provider tier)")
    agent_max_iter: int = Field(default=5, description="Max reasoning iterations per agent (lower = fewer LLM calls)")
    enable_memory: bool = Field(default=False, description="Crew memory (adds extra embedding/LLM calls per step)")
    timeout_seconds: int = Field(default=1800, description="Crew execution timeout in seconds")
    log_level: str = Field(default="INFO", description="Logging level")
    max_file_size_mb: int = Field(default=500, description="Max upload file size in MB")
    max_columns: int = Field(default=5000, description="Max dataset columns allowed")

    @property
    def resolved_model(self) -> str:
        if self.llm_model and self.llm_model != "":
            return self.llm_model
        return LLM_MODELS.get(self.llm_provider, LLM_MODELS["gemini"])


def get_settings() -> Settings:
    settings = Settings()
    # litellm/CrewAI read credentials from os.environ, not the Settings object.
    # Export what was loaded from .env so LLM calls authenticate. setdefault
    # never clobbers a real OS env var if one is already set.
    import os
    key_map = {
        "gemini": ("GEMINI_API_KEY", settings.gemini_api_key),
        "openai": ("OPENAI_API_KEY", settings.openai_api_key),
        "anthropic": ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
    }
    for env_name, value in key_map.values():
        if value:
            os.environ.setdefault(env_name, value)
    if settings.resolved_model:
        os.environ.setdefault("LLM_MODEL", settings.resolved_model)
    return settings
