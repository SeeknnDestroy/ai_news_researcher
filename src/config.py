from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


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

settings = Settings()


@dataclass
class InputData:
    urls: List[str]
    eval_enabled: bool = True


@dataclass
class CrawlItem:
    url: str
    text: str
    metadata: Dict[str, object]
    title: Optional[str] = None
    origin_url: Optional[str] = None


@dataclass
class SummaryItem:
    url: str
    origin_url: str
    source_name: str
    title: str
    date: Optional[date]
    date_inferred: bool
    summary_tr: str
    why_it_matters_tr: str
    tags: List[str]
    confidence: float


@dataclass
class ThemeGroup:
    name: str
    item_ids: List[int]


@dataclass
class ExcludedItem:
    url: str
    reason: str
