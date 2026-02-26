from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


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
