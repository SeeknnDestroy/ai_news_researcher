from __future__ import annotations

import asyncio
import re
from typing import Optional

from ..domain.contracts import DraftOutline, JudgeEvaluation, NewsletterSplitPayload, SummaryPayload
from ..domain.models import CrawlItem, ExcludedItem, NewsletterSplitResult, SummaryItem
from ..infrastructure.events import EventSink, NullEventSink, PipelineEvent
from ..infrastructure.llm_client import LLMClient, StructuredOutputError
from ..templates.prompts import (
    DRAFT_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    NEWSLETTER_SPLIT_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    THEME_REPORT_AGENT_SYSTEM_PROMPT,
    draft_agent_user_prompt,
    judge_agent_user_prompt,
    newsletter_split_user_prompt,
    summarize_user_prompt,
    theme_report_agent_user_prompt,
)
from ..utils import clamp_text_tokens, format_date, source_name_from_url


async def summarize_article_async(
    client: LLMClient,
    item: CrawlItem,
    *,
    audience: str = "mixed",
) -> SummaryItem:
    source_name = _source_name(item)
    title = item.title or "(Baslik yok)"
    article_text = clamp_text_tokens(item.text, 8000)
    user_prompt = summarize_user_prompt(audience, item.url, source_name, title, article_text)
    payload = await client.generate_structured(
        system=SUMMARIZE_SYSTEM_PROMPT,
        user=user_prompt,
        schema=SummaryPayload,
        task_name="article_summary",
    )
    return SummaryItem(
        url=item.url,
        origin_url=item.origin_url or item.url,
        source_name=payload.source_name or source_name,
        title=payload.title or title,
        date=None,
        date_inferred=False,
        summary_tr=payload.summary_tr,
        why_it_matters_tr=payload.why_it_matters_tr,
        tags=payload.tags,
        confidence=payload.confidence,
    )


async def split_newsletter_items_async(
    client: LLMClient,
    item: CrawlItem,
    *,
    max_items: int = 6,
    event_sink: EventSink | None = None,
) -> NewsletterSplitResult:
    sink = event_sink or NullEventSink()
    if not _is_newsletter(item.text, item.url):
        return NewsletterSplitResult(origin_url=item.url, items=[item], strategy="not_applicable")

    sink.emit(PipelineEvent(stage="SPLIT_NEWSLETTER", message=f"attempting origin={item.url}"))
    trimmed = clamp_text_tokens(_trim_newsletter_text(item.text), 8000)
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
        title = entry.get("title") or item.title or "Haber"
        url = entry.get("url") or _first_link(entry.get("text", "")) or item.url
        text = entry.get("text") or ""
        if not text.strip():
            continue
        results.append(
            CrawlItem(
                url=url,
                text=text.strip(),
                metadata={},
                title=str(title).strip(),
                origin_url=item.origin_url or item.url,
            )
        )

    if not results:
        return NewsletterSplitResult(
            origin_url=item.url,
            items=[item],
            strategy="not_split",
            validation_error=validation_error,
        )

    return NewsletterSplitResult(
        origin_url=item.url,
        items=results,
        strategy=strategy,
        validation_error=validation_error,
    )


async def generate_draft_outline(
    client: LLMClient,
    summaries: list[SummaryItem],
    *,
    critique: str = "",
    previous_draft: DraftOutline | None = None,
) -> DraftOutline:
    summaries_yaml = _summaries_yaml(summaries)
    previous_draft_str = previous_draft.model_dump_json(indent=2) if previous_draft else ""
    prompt = draft_agent_user_prompt(
        summaries_yaml=summaries_yaml,
        critique=critique,
        previous_draft=previous_draft_str,
    )
    return await client.generate_structured(
        system=DRAFT_AGENT_SYSTEM_PROMPT,
        user=prompt,
        schema=DraftOutline,
        task_name="draft_outline",
    )


async def evaluate_draft_outline(
    client: LLMClient,
    outline: DraftOutline,
    *,
    previous_critiques: str = "",
) -> JudgeEvaluation:
    prompt = judge_agent_user_prompt(outline.model_dump_json(indent=2), previous_critiques)
    return await client.generate_structured(
        system=JUDGE_AGENT_SYSTEM_PROMPT,
        user=prompt,
        schema=JudgeEvaluation,
        task_name="judge_evaluation",
    )


async def generate_final_report(
    client: LLMClient,
    outline: DraftOutline,
    summaries: list[SummaryItem],
    excluded: list[ExcludedItem],
    *,
    critique: str = "",
) -> str:
    report_parts: list[str] = []
    if outline.introduction_commentary:
        report_parts.append(f"# {outline.report_title}\n\n{outline.introduction_commentary}\n")
    else:
        report_parts.append(f"# {outline.report_title}\n")

    tasks = []
    for theme in outline.themes:
        theme_urls = {url for article in theme.articles for url in article.news_urls_included}
        theme_summaries = [item for item in summaries if item.url in theme_urls] or summaries
        prompt = theme_report_agent_user_prompt(
            theme_json=theme.model_dump_json(indent=2),
            summaries_yaml=_theme_summaries_yaml(theme_summaries),
            critique=critique,
        )
        tasks.append(
            client.generate_text(
                system=THEME_REPORT_AGENT_SYSTEM_PROMPT,
                user=prompt,
            )
        )

    for result in await asyncio.gather(*tasks):
        report_parts.append(result)
        report_parts.append("\n")

    report = "\n".join(report_parts)
    if excluded:
        report += "\n\n## Kullanilamayan Kaynaklar\n"
        for item in excluded:
            report += f"- {item.url} - {item.reason}\n"
    return report


def _source_name(item: CrawlItem) -> str:
    if item.metadata:
        for key in ("site_name", "og:site_name", "source"):
            if item.metadata.get(key):
                return str(item.metadata.get(key))
    return source_name_from_url(item.url)


def _summaries_yaml(summaries: list[SummaryItem]) -> str:
    lines: list[str] = []
    for item in summaries:
        lines.append(f"- URL: {item.url}")
        lines.append(f"  Title: {item.title}")
        lines.append(f"  Gelisme: {item.summary_tr}")
        lines.append(f"  Neden Onemli: {item.why_it_matters_tr}\n")
    return "\n".join(lines)


def _theme_summaries_yaml(summaries: list[SummaryItem]) -> str:
    lines: list[str] = []
    for item in summaries:
        lines.append(f"- URL: {item.url}")
        lines.append(f"  Source: {item.source_name}")
        lines.append(f"  Date: {format_date(item.date)}")
        lines.append(f"  Title: {item.title}")
        lines.append(f"  Gelisme: {item.summary_tr}")
        lines.append(f"  Neden Onemli: {item.why_it_matters_tr}\n")
    return "\n".join(lines)


def _is_newsletter(text: str, url: str) -> bool:
    if not text:
        return False
    normalized_text = text.lower().replace("’", "'")
    return "in today's edition of data points" in normalized_text and "deeplearning.ai" in url.lower()


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


def _find_marker(text: str, marker: str, start: int = 0) -> Optional[int]:
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
        match = re.search(re.escape(short_marker).replace(r"\ ", r"\s+"), text[start:], flags=re.IGNORECASE)
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
