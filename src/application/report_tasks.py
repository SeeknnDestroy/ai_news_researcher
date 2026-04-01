from __future__ import annotations

import asyncio

from ..domain.contracts import DraftOutline, FinalReportArticlePayload, JudgeEvaluation
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

    for theme in outline.themes:
        report_parts.append(f"## {theme.theme_name}\n")
        if theme.theme_commentary:
            report_parts.append(f"\n{theme.theme_commentary}\n")

        article_tasks = [
            _generate_article_section(
                client=client,
                article=article,
                summaries=summaries,
                theme_name=theme.theme_name,
                critique=critique,
            )
            for article in theme.articles
        ]

        for section in await asyncio.gather(*article_tasks):
            report_parts.append(section)
            report_parts.append("\n")

    report = "\n".join(report_parts)
    if excluded:
        report += "\n\n## Kullanilamayan Kaynaklar\n"
        for item in excluded:
            report += f"- {item.url} - {item.reason}\n"
    return report


async def _generate_article_section(
    *,
    client: LLMClient,
    article,
    summaries: list[SummaryItem],
    theme_name: str,
    critique: str,
) -> str:
    summary_map = {item.url: item for item in summaries}
    primary_summary = summary_map.get(article.primary_url)
    supporting_summaries = [summary_map[url] for url in article.news_urls_included if url != article.primary_url and url in summary_map]
    available_summaries = [item for item in [primary_summary, *supporting_summaries] if item is not None]
    fallback_summary = available_summaries[0] if available_summaries else summaries[0]
    if not available_summaries:
        supporting_summaries = [item for item in summaries if item.url != fallback_summary.url]
    prompt = theme_report_agent_user_prompt(
        theme_name=theme_name,
        article_json=article.model_dump_json(indent=2),
        primary_summary_yaml=_article_summary_yaml(primary_summary or fallback_summary),
        supporting_summaries_yaml=_theme_summaries_yaml(supporting_summaries),
        critique=critique,
    )
    article_payload = await client.generate_structured(
        system=THEME_REPORT_AGENT_SYSTEM_PROMPT,
        user=prompt,
        schema=FinalReportArticlePayload,
        task_name="final_report_theme",
    )
    source_summary = primary_summary or fallback_summary
    return "\n".join(
        [
            f"### <u>**{article.heading}**</u>",
            f"- **Tarih:** {format_date(source_summary.date)}",
            f"- **Kaynak:** [[{source_summary.source_name}]({source_summary.url})]",
            f"- **Gelişme:** {article_payload.gelisme}",
            f"- **Neden Önemli:** {article_payload.neden_onemli}",
        ]
    )


def _summaries_yaml(summaries: list[SummaryItem]) -> str:
    lines: list[str] = []
    for item in summaries:
        lines.append(f"- URL: {item.url}")
        lines.append(f"  Title: {item.title}")
        lines.append(f"  Gelisme: {item.summary_tr}")
        lines.append(f"  Neden Onemli: {item.why_it_matters_tr}\n")
    return "\n".join(lines)


def _theme_summaries_yaml(summaries: list[SummaryItem]) -> str:
    if not summaries:
        return "None\n"
    lines: list[str] = []
    for item in summaries:
        lines.append(f"- URL: {item.url}")
        lines.append(f"  Source: {item.source_name}")
        lines.append(f"  Date: {format_date(item.date)}")
        lines.append(f"  Title: {item.title}")
        lines.append(f"  Gelisme: {item.summary_tr}")
        lines.append(f"  Neden Onemli: {item.why_it_matters_tr}\n")
    return "\n".join(lines)


def _article_summary_yaml(summary: SummaryItem) -> str:
    return _theme_summaries_yaml([summary])
