from __future__ import annotations

import re
from datetime import date

import pytest

from src.agents.final_report_agent import generate_final_report as generate_final_report_agent
from src.application.ai_tasks import (
    evaluate_draft_outline as evaluate_draft_outline_task,
)
from src.application.ai_tasks import (
    generate_draft_outline as generate_draft_outline_task,
)
from src.application.ai_tasks import (
    generate_final_report as generate_final_report_task,
)
from src.domain.contracts import FinalReportArticlePayload, JudgeEvaluation, ThemeAssignmentPlan
from src.domain.models import SummaryItem


class FakeLegacyLLM:
    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        del system
        if task_name == "theme_assignment":
            story_unit_ids = re.findall(r'"story_unit_id":\s*"([^"]+)"', user)
            return schema.model_validate(
                ThemeAssignmentPlan(
                    report_title="Weekly",
                    themes=[
                        {
                            "theme_name": "Theme A",
                            "theme_commentary": "Theme intro",
                            "story_unit_ids": story_unit_ids,
                        }
                    ],
                )
            )
        if task_name == "judge_evaluation":
            return schema.model_validate(
                JudgeEvaluation(
                    critique="Looks good.",
                    specific_fixes_required=[],
                    passes_criteria=True,
                )
            )
        del user
        return schema.model_validate(
            FinalReportArticlePayload(
                gelisme="Gelisme metni.",
                neden_onemli="Neden onemli metni.",
            )
        )


def _summary_item(url: str, title: str, *, source_name: str = "Example") -> SummaryItem:
    return SummaryItem(
        url=url,
        origin_url=url,
        source_name=source_name,
        title=title,
        date=date(2026, 3, 11),
        date_inferred=False,
        summary_tr=f"{title} ozet metni.",
        why_it_matters_tr=f"{title} neden onemli.",
        tags=["ai"],
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_ai_tasks_generate_final_report_remains_awaitable():
    outline = {
        "report_title": "Weekly",
        "themes": [
            {
                "theme_name": "Theme A",
                "theme_commentary": "Theme intro",
                "articles": [
                    {
                        "heading": "Heading",
                        "primary_url": "https://first",
                        "news_urls_included": ["https://first"],
                        "content_plan": "Plan",
                    }
                ],
            }
        ],
    }

    report = await generate_final_report_task(
        FakeLegacyLLM(),
        outline,
        [_summary_item("https://first", "First Title")],
        [],
        critique="",
    )

    assert "### <u>**Heading**</u>" in report
    assert "[[Example](https://first)]" in report


@pytest.mark.asyncio
async def test_ai_tasks_legacy_outline_wrappers_remain_awaitable():
    summaries = [_summary_item("https://first", "First Title")]

    outline = await generate_draft_outline_task(
        FakeLegacyLLM(),
        summaries,
        critique="",
    )
    evaluation = await evaluate_draft_outline_task(FakeLegacyLLM(), outline, previous_critiques="")

    assert outline.report_title == "Weekly"
    assert outline.themes[0].articles[0].primary_url == "https://first"
    assert evaluation.passes_criteria is True


@pytest.mark.asyncio
async def test_final_report_agent_falls_back_when_outline_primary_url_is_missing():
    outline = {
        "report_title": "Weekly",
        "themes": [
            {
                "theme_name": "Theme A",
                "theme_commentary": "Theme intro",
                "articles": [
                    {
                        "heading": "Heading",
                        "primary_url": "https://missing",
                        "news_urls_included": ["https://missing"],
                        "content_plan": "Plan",
                    }
                ],
            }
        ],
    }
    summaries = [
        _summary_item("https://first", "First Title", source_name="First Source"),
        _summary_item("https://second", "Second Title", source_name="Second Source"),
    ]

    report = await generate_final_report_agent(None, outline, summaries, [], critique="")

    assert "### <u>**Heading**</u>" in report
    assert "**Kaynak:** [[First Source](https://first)]" in report
    assert "**Tarih:** 11 March 2026" in report
