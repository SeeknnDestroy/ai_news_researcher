from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.4-nano"
    openai_temperature: float = 0.2
    openai_timeout_s: int = 60
    openai_reasoning_effort: str = "none"
    openai_verbosity: str = "low"
    openai_max_output_tokens: int = 2000
    openai_rpm_limit: int = 500
    openai_tpm_limit: int = 200_000
    openai_tpd_limit: int = 2_000_000

    eval_revision_threshold: float = 75.0
    eval_groundedness_threshold: float = 70.0
    eval_coverage_threshold: float = 70.0
    eval_default_trials: int = 2

    max_report_chars: int = 12000
    max_factcheck_items: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
