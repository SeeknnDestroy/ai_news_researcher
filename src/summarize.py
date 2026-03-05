from __future__ import annotations

from .config import CrawlItem, SummaryItem
from .llm import XAIConfig, generate_json, generate_json_async
from .utils import clamp_text_tokens, source_name_from_url
from .templates.prompts import SUMMARIZE_SYSTEM_PROMPT, summarize_user_prompt


def summarize_article(config: XAIConfig, item: CrawlItem, audience: str = "mixed") -> SummaryItem:
    source_name = _source_name(item)
    title = item.title or "(Başlık yok)"
    article_text = clamp_text_tokens(item.text, 8000)

    user_prompt = summarize_user_prompt(audience, item.url, source_name, title, article_text)

    data = generate_json(config=config, system=SUMMARIZE_SYSTEM_PROMPT, user=user_prompt)

    return SummaryItem(
        url=item.url,
        origin_url=item.origin_url or item.url,
        source_name=str(data.get("source_name") or source_name),
        title=str(data.get("title") or title),
        date=None,  # will be set later
        date_inferred=False,
        summary_tr=str(data.get("summary_tr", "")).strip(),
        why_it_matters_tr=str(data.get("why_it_matters_tr", "")).strip(),
        tags=_safe_tags(data.get("tags")),
        confidence=_clamp_confidence(_safe_float(data.get("confidence", 0.0))),
    )


async def summarize_article_async(config: XAIConfig, item: CrawlItem, audience: str = "mixed") -> SummaryItem | None:
    source_name = _source_name(item)
    title = item.title or "(Başlık yok)"
    article_text = clamp_text_tokens(item.text, 8000)

    user_prompt = summarize_user_prompt(audience, item.url, source_name, title, article_text)

    try:
        data = await generate_json_async(config=config, system=SUMMARIZE_SYSTEM_PROMPT, user=user_prompt)
    except Exception:
        return None

    return SummaryItem(
        url=item.url,
        origin_url=item.origin_url or item.url,
        source_name=str(data.get("source_name") or source_name),
        title=str(data.get("title") or title),
        date=None,  # will be set later
        date_inferred=False,
        summary_tr=str(data.get("summary_tr", "")).strip(),
        why_it_matters_tr=str(data.get("why_it_matters_tr", "")).strip(),
        tags=_safe_tags(data.get("tags")),
        confidence=_clamp_confidence(_safe_float(data.get("confidence", 0.0))),
    )


def _source_name(item: CrawlItem) -> str:
    if item.metadata:
        for key in ("site_name", "og:site_name", "source"):
            if item.metadata.get(key):
                return str(item.metadata.get(key))
    return source_name_from_url(item.url)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for tag in value:
        text = str(tag).strip().lower()
        if not text:
            continue
        cleaned.append(text[:30])
    return cleaned[:5]


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
