from __future__ import annotations

import re

from ..date_extract import extract_date
from ..domain.contracts import NewsletterSplitPayload, StoryCardPayload
from ..domain.models import CrawlItem, NewsletterSplitResult, StoryCard, SummaryItem
from ..infrastructure.events import EventSink, NullEventSink, PipelineEvent
from ..infrastructure.llm_client import LLMClient, StructuredOutputError
from ..templates.prompts import (
    NEWSLETTER_SPLIT_SYSTEM_PROMPT,
    STORY_CARD_SYSTEM_PROMPT,
    newsletter_split_user_prompt,
    story_card_user_prompt,
)
from ..utils import clamp_text_tokens, source_family_from_url, source_name_from_url


async def extract_story_card_async(client: LLMClient, item: CrawlItem) -> StoryCard:
    prepared_item = prepare_crawl_item(item)
    article_text = clamp_text_tokens(prepared_item.text, 8000)
    user_prompt = story_card_user_prompt(prepared_item, article_text)
    payload = await client.generate_structured(
        system=STORY_CARD_SYSTEM_PROMPT,
        user=user_prompt,
        schema=StoryCardPayload,
        task_name="story_card_extraction",
    )
    return StoryCard(
        url=prepared_item.url,
        origin_url=prepared_item.origin_url or prepared_item.url,
        source_name=prepared_item.source_name or source_name_from_url(prepared_item.url),
        title_raw=prepared_item.title_raw or prepared_item.title or "(Baslik yok)",
        published_at=prepared_item.published_at,
        published_at_inferred=prepared_item.published_at_inferred,
        raw_text=prepared_item.text,
        content_type=prepared_item.content_type,
        crawl_quality_flags=list(prepared_item.crawl_quality_flags),
        blocked_or_partial=prepared_item.blocked_or_partial,
        source_family=prepared_item.source_family or source_family_from_url(prepared_item.url),
        story_title_tr=payload.story_title_tr,
        story_type=payload.story_type,
        key_facts=payload.key_facts,
        must_keep_entities=payload.must_keep_entities,
        must_keep_facts=payload.must_keep_facts,
        why_it_matters_tr=payload.why_it_matters_tr,
        technical_relevance=payload.technical_relevance,
        strategic_relevance=payload.strategic_relevance,
        confidence=payload.confidence,
    )


async def summarize_article_async(
    client: LLMClient,
    item: CrawlItem,
    *,
    audience: str = "mixed",
) -> SummaryItem:
    del audience
    story_card = await extract_story_card_async(client, item)
    summary_text = " ".join(story_card.key_facts[:3]).strip() or story_card.story_title_tr
    return SummaryItem(
        url=story_card.url,
        origin_url=story_card.origin_url,
        source_name=story_card.source_name,
        title=story_card.story_title_tr,
        date=story_card.published_at,
        date_inferred=story_card.published_at_inferred,
        summary_tr=summary_text,
        why_it_matters_tr=story_card.why_it_matters_tr,
        tags=[],
        confidence=story_card.confidence,
    )


async def split_newsletter_items_async(
    client: LLMClient,
    item: CrawlItem,
    *,
    max_items: int = 6,
    event_sink: EventSink | None = None,
) -> NewsletterSplitResult:
    sink = event_sink or NullEventSink()
    prepared_item = prepare_crawl_item(item)
    if not _is_newsletter(prepared_item.text, prepared_item.url):
        return NewsletterSplitResult(
            origin_url=prepared_item.url, items=[prepared_item], strategy="not_applicable"
        )

    sink.emit(
        PipelineEvent(stage="SPLIT_NEWSLETTER", message=f"attempting origin={prepared_item.url}")
    )
    trimmed = clamp_text_tokens(_trim_newsletter_text(prepared_item.text), 8000)
    validation_error: str | None = None

    try:
        payload = await client.generate_structured(
            system=NEWSLETTER_SPLIT_SYSTEM_PROMPT,
            user=newsletter_split_user_prompt(max_items, trimmed),
            schema=NewsletterSplitPayload,
            task_name="newsletter_split",
        )
        segments = _segments_from_markers(trimmed, payload.items, max_items)
        strategy = "llm_markers"
    except StructuredOutputError as exc:
        validation_error = str(exc)
        segments = []
        strategy = "heuristic_fallback"

    if not segments:
        segments = _extract_items(trimmed, max_items)
        if strategy != "heuristic_fallback":
            strategy = "heuristic_fallback"

    results: list[CrawlItem] = []
    for entry in segments:
        title = entry.get("title") or prepared_item.title or "Haber"
        url = entry.get("url") or _first_link(entry.get("text", "")) or prepared_item.url
        text = entry.get("text") or ""
        if not text.strip():
            continue
        split_item = CrawlItem(
            url=url,
            text=text.strip(),
            metadata={},
            title=str(title).strip(),
            origin_url=prepared_item.origin_url or prepared_item.url,
            title_raw=str(title).strip(),
            source_name=prepared_item.source_name,
            source_family=prepared_item.source_family,
            published_at=prepared_item.published_at,
            published_at_inferred=prepared_item.published_at_inferred,
            content_type=prepared_item.content_type,
            crawl_quality_flags=list(prepared_item.crawl_quality_flags),
            blocked_or_partial=prepared_item.blocked_or_partial,
        )
        results.append(prepare_crawl_item(split_item))

    if not results:
        return NewsletterSplitResult(
            origin_url=prepared_item.url,
            items=[prepared_item],
            strategy="not_split",
            validation_error=validation_error,
        )

    return NewsletterSplitResult(
        origin_url=prepared_item.url,
        items=results,
        strategy=strategy,
        validation_error=validation_error,
    )


def prepare_crawl_item(item: CrawlItem) -> CrawlItem:
    source_name = item.source_name or _source_name(item)
    source_family = item.source_family or source_family_from_url(item.url)
    title_raw = item.title_raw or item.title or "(Baslik yok)"
    content_type = item.content_type or str(item.metadata.get("content_type") or "text/html")
    crawl_quality_flags = _normalized_quality_flags(item)
    date_value = item.published_at
    date_inferred = item.published_at_inferred
    if date_value is None:
        date_result = extract_date(item.metadata, item.text, item.url)
        date_value = date_result.value
        date_inferred = date_result.inferred

    return CrawlItem(
        url=item.url,
        text=item.text,
        metadata=dict(item.metadata),
        title=item.title or title_raw,
        origin_url=item.origin_url or item.url,
        title_raw=title_raw,
        source_name=source_name,
        source_family=source_family,
        published_at=date_value,
        published_at_inferred=date_inferred,
        content_type=content_type,
        crawl_quality_flags=crawl_quality_flags,
        blocked_or_partial=item.blocked_or_partial or _is_partial(crawl_quality_flags),
    )


def _normalized_quality_flags(item: CrawlItem) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for value in list(item.crawl_quality_flags) + _inferred_quality_flags(item):
        normalized_value = str(value).strip().lower()
        if not normalized_value:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        flags.append(normalized_value)
    return flags


def _inferred_quality_flags(item: CrawlItem) -> list[str]:
    flags: list[str] = []
    if not item.text.strip():
        flags.append("empty")
    if len(item.text.split()) < 80:
        flags.append("short_text")
    if item.metadata.get("final_url"):
        flags.append("redirected")
    return flags


def _is_partial(flags: list[str]) -> bool:
    return any(flag in {"blocked", "empty"} for flag in flags)


def _source_name(item: CrawlItem) -> str:
    if item.source_name:
        return item.source_name
    if item.metadata:
        for key in ("site_name", "og:site_name", "source"):
            if item.metadata.get(key):
                return str(item.metadata.get(key))
    return source_name_from_url(item.url)


def _is_newsletter(text: str, url: str) -> bool:
    if not text:
        return False
    normalized_text = text.lower().replace("’", "'")
    return (
        "in today's edition of data points" in normalized_text and "deeplearning.ai" in url.lower()
    )


def _trim_newsletter_text(text: str) -> str:
    if not text:
        return ""
    stop_markers = [
        "want to know more",
        "a special offer",
        "subscribe",
        "try pro membership",
        "enroll now",
        "special offer",
        "subscribe to data points",
    ]
    lowered = text.lower()
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def _segments_from_markers(text: str, items, max_items: int) -> list[dict]:
    if not items:
        return []

    found: list[dict] = []
    for entry in items:
        if len(found) >= max_items:
            break
        start_idx = _find_marker(text, entry.start_marker)
        if start_idx is None:
            continue
        found.append(
            {
                "title": entry.title,
                "url": entry.url,
                "start_idx": start_idx,
                "end_marker": entry.end_marker,
            }
        )

    if not found:
        return []

    found.sort(key=lambda entry: entry["start_idx"])
    total_len = len(text)
    segments: list[dict] = []
    for idx, entry in enumerate(found):
        start_idx = entry["start_idx"]
        end_idx = None
        if entry["end_marker"]:
            end_idx = _find_marker(text, entry["end_marker"], start=start_idx + 1)
        if end_idx is None:
            next_start = found[idx + 1]["start_idx"] if idx + 1 < len(found) else total_len
            end_idx = next_start
        if end_idx <= start_idx:
            continue
        block_text = text[start_idx:end_idx].strip()
        if len(block_text.split()) < 25:
            continue
        segments.append(
            {
                "title": entry["title"] or block_text.splitlines()[0].strip(),
                "url": entry["url"],
                "text": block_text,
            }
        )
        if len(segments) >= max_items:
            break
    return segments


def _find_marker(text: str, marker: str, start: int = 0) -> int | None:
    if not marker:
        return None

    pattern = re.escape(marker).replace(r"\ ", r"\s+")
    match = re.search(pattern, text[start:], flags=re.IGNORECASE)
    if match:
        return start + match.start()

    normalized_text = text[start:].replace("’", "'").replace("“", '"').replace("”", '"')
    normalized_marker = marker.replace("’", "'").replace("“", '"').replace("”", '"')
    pattern = re.escape(normalized_marker).replace(r"\ ", r"\s+")
    match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
    if match:
        return start + match.start()

    words = marker.split()
    if len(words) > 8:
        short_marker = " ".join(words[:8])
        match = re.search(
            re.escape(short_marker).replace(r"\ ", r"\s+"), text[start:], flags=re.IGNORECASE
        )
        if match:
            return start + match.start()
    return None


def _extract_items(text: str, max_items: int) -> list[dict]:
    if not text:
        return []

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    headings = list(re.finditer(r"\*\*(.+?)\*\*", text))
    items: list[dict] = []

    if headings:
        for idx, match in enumerate(headings):
            title = match.group(1).strip()
            start = match.end()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            items.append({"title": title, "url": _first_link(body), "text": body})
            if len(items) >= max_items:
                break
        return items

    for block in blocks:
        if len(items) >= max_items:
            break
        if len(block.split()) < 30:
            continue
        title = block.split("\n", 1)[0].strip()
        items.append({"title": title, "url": _first_link(block), "text": block})
    return items


def _first_link(text: str) -> str:
    markdown_match = re.search(r"\((https?://[^)]+)\)", text)
    if markdown_match:
        return markdown_match.group(1)
    plain_match = re.search(r"(https?://\S+)", text)
    return plain_match.group(1) if plain_match else ""
