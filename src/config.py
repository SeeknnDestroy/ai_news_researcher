from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain.models import CrawlItem, ExcludedItem, InputData, SummaryItem, ThemeGroup


class Settings(BaseSettings):
    xai_api_key: Optional[str] = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_model: str = "grok-4-1-fast-reasoning"
    xai_temperature: float = 0.2
    xai_timeout_s: int = 60
    
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
