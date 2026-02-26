from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Dict, Iterable, Optional, Tuple

import dateparser
from dateparser.search import search_dates


@dataclass
class DateResult:
    value: Optional[date]
    inferred: bool
    source: Optional[str]


META_DATE_KEYS = [
    "article:published_time",
    "og:published_time",
    "published_time",
    "pubdate",
    "publish_date",
    "date",
    "published",
    "article:modified_time",
    "og:updated_time",
]


def extract_date(metadata: Dict[str, object], text: str, url: str) -> DateResult:
    meta_date = _extract_from_metadata(metadata)
    if meta_date:
        return DateResult(meta_date, inferred=False, source="metadata")

    text_date = _extract_from_text(text)
    if text_date:
        return DateResult(text_date, inferred=True, source="text")

    url_date = _extract_from_url(url)
    if url_date:
        return DateResult(url_date, inferred=True, source="url")

    return DateResult(None, inferred=True, source=None)


def is_within_window(value: date, start: date, end: date) -> bool:
    lower = start - timedelta(days=1)
    upper = end + timedelta(days=1)
    return lower <= value <= upper


def _extract_from_metadata(metadata: Dict[str, object]) -> Optional[date]:
    for key in META_DATE_KEYS:
        raw = metadata.get(key)
        parsed = _parse_date_value(raw)
        if parsed:
            return parsed

    # Try any other metadata value that looks like a date
    for _, raw in metadata.items():
        parsed = _parse_date_value(raw)
        if parsed:
            return parsed

    return None


def _extract_from_text(text: str) -> Optional[date]:
    if not text:
        return None

    sample = text[:2000]
    results = search_dates(sample, languages=["en", "tr"])
    if results:
        # Prefer the first date found in the text
        _, dt = results[0]
        return dt.date()

    # Fallback regex for ISO-like patterns
    match = re.search(r"\b(20\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})\b", sample)
    if match:
        year, month, day = map(int, match.groups())
        return date(year, month, day)

    return None


def _extract_from_url(url: str) -> Optional[date]:
    match = re.search(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", url)
    if match:
        year, month, day = map(int, match.groups())
        return date(year, month, day)

    match = re.search(r"(20\d{2})(\d{2})(\d{2})", url)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def _parse_date_value(raw: object) -> Optional[date]:
    if raw is None:
        return None

    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw

    if isinstance(raw, datetime):
        return raw.date()

    if isinstance(raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(raw).date()
        except (ValueError, OSError):
            return None

    if isinstance(raw, str):
        parsed = dateparser.parse(raw)
        if parsed:
            return parsed.date()

    return None
