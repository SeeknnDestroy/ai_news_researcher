from __future__ import annotations

from datetime import date

import pytest

from src.application.ai_tasks import generate_final_report, split_newsletter_items_async
from src.application.report_workflow import ReportWorkflowService, validate_outline_structure
from src.domain.contracts import (
    DraftOutline,
    DraftOutlineArticle,
    DraftOutlineTheme,
    FinalReportArticlePayload,
    JudgeEvaluation,
    NewsletterSplitItemPayload,
    NewsletterSplitPayload,
)
from src.domain.models import CrawlItem, ExcludedItem, SummaryItem
from src.infrastructure.llm_client import StructuredOutputError


class FakeLLM:
    def __init__(self, *, structured: dict[str, list] | None = None, text: list[str] | None = None) -> None:
        self._structured = {key: list(value) for key, value in (structured or {}).items()}
        self._text = list(text or [])
        self.generated_prompts: list[str] = []
        self.generated_structured_prompts: list[tuple[str, str]] = []

    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        self.generated_structured_prompts.append((task_name, user))
        del system, user
        try:
            result = self._structured[task_name].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"unexpected structured task: {task_name}") from exc
        if isinstance(result, Exception):
            raise result
        return result

    async def generate_text(self, *, system: str, user: str) -> str:
        del system
        self.generated_prompts.append(user)
        try:
            return self._text.pop(0)
        except IndexError as exc:
            raise AssertionError("unexpected text generation call") from exc


class NeverCalledLLM:
    async def generate_structured(self, *args, **kwargs):  # pragma: no cover - not used
        raise AssertionError("should not call generate_structured")

    async def generate_text(self, *args, **kwargs):  # pragma: no cover - not used
        raise AssertionError("should not call generate_text")


def _newsletter_item(url: str, text: str) -> CrawlItem:
    return CrawlItem(url=url, text=text, metadata={}, title="newsletter", origin_url=url)


@pytest.mark.asyncio
async def test_split_newsletter_not_applicable_does_not_call_llm():
    item = _newsletter_item("https://example.com/article", "just a regular article text")
    result = await split_newsletter_items_async(NeverCalledLLM(), item)
    assert result.strategy == "not_applicable"
    assert result.items == [item]



@pytest.mark.asyncio
async def test_split_newsletter_falls_back_to_heuristic_when_llm_errors():
    text = """
In today's edition of Data Points

**First Item**
A good block for the heuristic fallback.

**Second Item**
A second block with plenty of words so it is treated as valid.
""".strip()
    llm = FakeLLM(structured={"newsletter_split": [StructuredOutputError("invalid markers")]})
    result = await split_newsletter_items_async(llm, _newsletter_item("https://deeplearning.ai/the-batch", text))
    assert result.strategy == "heuristic_fallback"
    assert "invalid markers" in (result.validation_error or "")
    assert len(result.items) >= 2


@pytest.mark.asyncio
async def test_split_newsletter_returns_not_split_when_no_segments():
    text = """
In today's edition of Data Points

Short content
""".strip()
    llm = FakeLLM(structured={"newsletter_split": [StructuredOutputError("invalid markers")]})
    result = await split_newsletter_items_async(llm, _newsletter_item("https://deeplearning.ai/the-batch", text))
    assert result.strategy == "not_split"
    assert len(result.items) == 1
    assert result.items[0].url == "https://deeplearning.ai/the-batch"


def _summary_item(url: str, title: str) -> SummaryItem:
    return SummaryItem(
        url=url,
        origin_url=url,
        source_name="source",
        title=title,
        date=date(2026, 3, 11),
        date_inferred=False,
        summary_tr=f"summary for {title}",
        why_it_matters_tr=f"why it matters for {title}",
        tags=["ai"],
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_generate_final_report_falls_back_to_all_summaries_and_appends_excluded():
    summaries = [_summary_item("https://first", "First Title"), _summary_item("https://second", "Second Title")]
    theme = DraftOutlineTheme(
        theme_name="Theme A",
        theme_commentary="Theme intro",
        articles=[
            DraftOutlineArticle(
                heading="Heading",
                primary_url="https://missing",
                news_urls_included=["https://missing"],
                content_plan="Plan",
            )
        ],
    )
    outline = DraftOutline(report_title="Weekly", introduction_commentary="Intro", themes=[theme])
    excluded = [ExcludedItem(url="https://excluded", reason="no signal")]
    llm = FakeLLM(structured={"final_report_theme": [FinalReportArticlePayload(gelisme="Gelisme", neden_onemli="Onemli")]})
    report = await generate_final_report(llm, outline, summaries, excluded)
    assert "First Title" in llm.generated_structured_prompts[0][1]
    assert "Second Title" in llm.generated_structured_prompts[0][1]
    assert "## Kullanilamayan Kaynaklar" in report
    assert "https://excluded" in report


def _draft_outline() -> DraftOutline:
    theme = DraftOutlineTheme(
        theme_name="Theme",
        theme_commentary="overview",
        articles=[
            DraftOutlineArticle(
                heading="Heading",
                primary_url="https://first",
                news_urls_included=["https://first"],
                content_plan="Plan",
            )
        ],
    )
    return DraftOutline(report_title="Weekly", introduction_commentary="Intro", themes=[theme])


@pytest.mark.asyncio
async def test_report_workflow_skips_revisions_when_eval_disabled():
    llm = FakeLLM(
        structured={
            "draft_outline": [_draft_outline()],
            "judge_evaluation": [JudgeEvaluation(critique="ok", specific_fixes_required=[], passes_criteria=True)],
            "final_report_theme": [FinalReportArticlePayload(gelisme="Gelisme", neden_onemli="Onemli")],
        },
    )
    service = ReportWorkflowService(llm_client=llm)
    results = await service.run(summaries=[_summary_item("https://first", "First Title")], excluded=[], eval_enabled=False)
    assert results.revision_count == 0
    assert results.critique_history == []
    assert results.evaluation.passes_criteria


@pytest.mark.asyncio
async def test_report_workflow_retries_once_when_eval_enabled():
    llm = FakeLLM(
        structured={
            "draft_outline": [_draft_outline(), _draft_outline()],
            "judge_evaluation": [
                JudgeEvaluation(critique="needs work", specific_fixes_required=["move faster"], passes_criteria=False),
                JudgeEvaluation(critique="now stable", specific_fixes_required=[], passes_criteria=True),
            ],
            "final_report_theme": [FinalReportArticlePayload(gelisme="Gelisme", neden_onemli="Onemli")],
        },
    )
    service = ReportWorkflowService(llm_client=llm)
    results = await service.run(summaries=[_summary_item("https://first", "First Title")], excluded=[], eval_enabled=True)
    assert results.revision_count == 1
    assert len(results.critique_history) == 1
    assert "move faster" in results.critique_history[0]


@pytest.mark.asyncio
async def test_generate_final_report_uses_primary_url_for_source_and_date():
    primary = _summary_item("https://primary", "Primary Title")
    supporting = _summary_item("https://supporting", "Supporting Title")
    supporting.source_name = "supporting-source"
    supporting.date = date(2026, 3, 12)
    theme = DraftOutlineTheme(
        theme_name="Theme A",
        articles=[
            DraftOutlineArticle(
                heading="Heading",
                primary_url="https://primary",
                news_urls_included=["https://primary", "https://supporting"],
                content_plan="Plan",
            )
        ],
    )
    outline = DraftOutline(report_title="Weekly", themes=[theme])
    llm = FakeLLM(structured={"final_report_theme": [FinalReportArticlePayload(gelisme="Gelisme", neden_onemli="Onemli")]})

    report = await generate_final_report(llm, outline, [primary, supporting], [])

    assert "**Kaynak:** [[source](https://primary)]" in report
    assert "**Tarih:** 11 March 2026" in report
    prompt_text = llm.generated_structured_prompts[0][1]
    assert "Primary Summary:" in prompt_text
    assert "Supporting Summaries:" in prompt_text
    assert "Supporting Title" in prompt_text


def test_validate_outline_structure_rejects_over_grouped_outline():
    summaries = [_summary_item(f"https://example.com/{index}", f"Title {index}") for index in range(28)]
    articles = []
    for index in range(10):
        start = index * 2
        article_urls = [item.url for item in summaries[start : start + 2]]
        articles.append(
            DraftOutlineArticle.model_construct(
                heading=f"Heading {index}",
                primary_url=article_urls[0],
                news_urls_included=article_urls,
                content_plan="Plan",
            )
        )
    outline = DraftOutline(
        report_title="Weekly",
        themes=[DraftOutlineTheme(theme_name="1. Theme", articles=articles)],
    )

    errors = validate_outline_structure(outline, summaries)

    assert any("at least" in error for error in errors)
    assert any("multi-URL article blocks" in error for error in errors)


@pytest.mark.asyncio
async def test_report_workflow_retries_when_outline_validation_fails():
    bad_outline = DraftOutline(
        report_title="Weekly",
        themes=[
            DraftOutlineTheme(
                theme_name="1. Theme",
                articles=[
                    DraftOutlineArticle(
                        heading="Heading",
                        primary_url="https://first",
                        news_urls_included=["https://first"],
                        content_plan="Plan",
                    )
                ],
            )
        ],
    )
    good_outline = DraftOutline(
        report_title="Weekly",
        themes=[
            DraftOutlineTheme(
                theme_name="1. Theme",
                articles=[
                    DraftOutlineArticle(
                        heading="Heading One",
                        primary_url="https://first",
                        news_urls_included=["https://first"],
                        content_plan="Plan",
                    ),
                    DraftOutlineArticle(
                        heading="Heading Two",
                        primary_url="https://second",
                        news_urls_included=["https://second"],
                        content_plan="Plan",
                    ),
                ],
            )
        ],
    )
    llm = FakeLLM(
        structured={
            "draft_outline": [bad_outline, good_outline],
            "judge_evaluation": [
                JudgeEvaluation(critique="judge ok", specific_fixes_required=[], passes_criteria=True),
                JudgeEvaluation(critique="judge ok", specific_fixes_required=[], passes_criteria=True),
            ],
            "final_report_theme": [
                FinalReportArticlePayload(gelisme="Gelisme bir", neden_onemli="Onemli bir"),
                FinalReportArticlePayload(gelisme="Gelisme iki", neden_onemli="Onemli iki"),
            ],
        },
    )
    service = ReportWorkflowService(llm_client=llm)
    summaries = [_summary_item("https://first", "First Title"), _summary_item("https://second", "Second Title")]

    results = await service.run(summaries=summaries, excluded=[], eval_enabled=True)

    assert results.revision_count == 1
    assert any("Deterministic outline validation failed" in item for item in results.critique_history)
