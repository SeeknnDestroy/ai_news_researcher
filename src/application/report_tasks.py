from __future__ import annotations

import asyncio

from ..domain.contracts import DraftOutline, JudgeEvaluation
from ..domain.models import ExcludedItem, SummaryItem
from ..infrastructure.llm_client import LLMClient
from ..templates.prompts import (
    DRAFT_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    THEME_REPORT_AGENT_SYSTEM_PROMPT,
    draft_agent_user_prompt,
    judge_agent_user_prompt,
    theme_report_agent_user_prompt,
)
from ..utils import format_date


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
