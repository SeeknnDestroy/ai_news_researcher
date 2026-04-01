from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from src.application.pipeline import PipelineRequest, PipelineRunner
from src.domain.models import (
    CrawlFailure,
    CrawlItem,
    CrawlStageResult,
    ExcludedItem,
    NewsletterSplitResult,
)
from src.infrastructure.events import NullEventSink
from src.infrastructure.llm_client import StructuredOutputError
from src.infrastructure.persistence import FileSystemPipelineStore
from src.storage_paths import dated_input_path


@dataclass
class FakeCrawlService:
    result: CrawlStageResult

    async def crawl(self, urls: list[str], *, max_concurrency: int) -> CrawlStageResult:
        assert urls
        assert max_concurrency >= 1
        return self.result


class FakeLLMClient:
    def __init__(self, *, structured: dict[str, list], text: list[str] | None = None) -> None:
        self.structured = {key: list(value) for key, value in structured.items()}
        self.text = list(text or [])
        self.usage_snapshot = {
            "totals": {
                "request_count": 4,
                "input_tokens": 430,
                "output_tokens": 170,
                "cached_input_tokens": 80,
                "reasoning_tokens": 12,
                "total_tokens": 600,
            },
            "by_task": {
                "article_summary": {
                    "request_count": 1,
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 160,
                },
                "draft_outline": {
                    "request_count": 1,
                    "input_tokens": 90,
                    "output_tokens": 35,
                    "cached_input_tokens": 10,
                    "reasoning_tokens": 4,
                    "total_tokens": 125,
                },
                "judge_evaluation": {
                    "request_count": 1,
                    "input_tokens": 70,
                    "output_tokens": 25,
                    "cached_input_tokens": 15,
                    "reasoning_tokens": 3,
                    "total_tokens": 95,
                },
                "final_report_theme": {
                    "request_count": 1,
                    "input_tokens": 150,
                    "output_tokens": 70,
                    "cached_input_tokens": 35,
                    "reasoning_tokens": 5,
                    "total_tokens": 220,
                },
            },
            "by_role": {
                "researcher": {
                    "request_count": 1,
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 160,
                },
                "writer": {
                    "request_count": 2,
                    "input_tokens": 240,
                    "output_tokens": 105,
                    "cached_input_tokens": 45,
                    "reasoning_tokens": 9,
                    "total_tokens": 345,
                },
                "judge": {
                    "request_count": 1,
                    "input_tokens": 70,
                    "output_tokens": 25,
                    "cached_input_tokens": 15,
                    "reasoning_tokens": 3,
                    "total_tokens": 95,
                },
            },
        }

    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        del system, user
        try:
            value = self.structured[task_name].pop(0)
        except (KeyError, IndexError) as exc:
            if task_name == "final_report_theme":
                return schema.model_validate(
                    {
                        "gelisme": "Theme development summary.",
                        "neden_onemli": "Theme strategic implication.",
                    }
                )
            raise AssertionError(f"unexpected structured task: {task_name}") from exc
        if isinstance(value, Exception):
            raise value
        return schema.model_validate(value)

    async def generate_text(self, *, system: str, user: str) -> str:
        del system, user
        try:
            value = self.text.pop(0)
        except IndexError as exc:
            raise AssertionError("unexpected text generation call") from exc
        if isinstance(value, Exception):
            raise value
        return value

    def get_usage_summary(self):
        return self.usage_snapshot


def _write_input(base_dir: Path, target_date: date, urls: list[str], evaluation: bool = True) -> Path:
    input_path = dated_input_path(target_date, base_dir=base_dir)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "evaluation: " + ("true" if evaluation else "false") + "\n"
        + "urls:\n"
        + "".join(f'  - "{url}"\n' for url in urls),
        encoding="utf-8",
    )
    return input_path


def _outline(urls: list[str]) -> dict:
    return {
        "report_title": "Weekly AI Report",
        "introduction_commentary": "Intro commentary.",
        "themes": [
            {
                "theme_name": "1. Agentic Delivery",
                "theme_commentary": "Theme overview.",
                "articles": [
                    {
                        "heading": "Execution speed improves",
                        "primary_url": urls[0],
                        "news_urls_included": urls,
                        "content_plan": "Cover the engineering and delivery signals.",
                    }
                ],
            }
        ],
    }


def _summary(title: str, source_name: str = "Example Source") -> dict:
    return {
        "title": title,
        "source_name": source_name,
        "summary_tr": "Teknik gelisme ekiplerin teslim hizini artiriyor ve olculebilir fayda sagliyor.",
        "why_it_matters_tr": "Bu gelisme SDLC verimliligini ve uygulama guvenilirligini etkiliyor.",
        "tags": ["agentic", "sdlc"],
        "confidence": 0.81,
    }


@pytest.mark.asyncio
async def test_pipeline_runner_persists_successful_run(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Story A"), _summary("Story B")],
            "draft_outline": [_outline(urls)],
            "judge_evaluation": [
                {
                    "critique": "Good structure.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(url=urls[0], text="Article text A", metadata={}, title="Story A", origin_url=urls[0]),
                CrawlItem(url=urls[1], text="Article text B", metadata={}, title="Story B", origin_url=urls[1]),
            ],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2))

    assert result.persistence.report_path == tmp_path / "reports" / "2026-03" / "13-03-2026_weekly.md"
    assert result.persistence.report_path.exists()
    assert result.persistence.artifact_path.exists()
    assert result.metadata.retries["draft_revision"] == 0
    artifact_content = result.persistence.artifact_path.read_text(encoding="utf-8")
    assert '"stage_timings"' in artifact_content
    assert '"report_title": "Weekly AI Report"' in artifact_content
    assert '"llm_usage"' in artifact_content
    assert '"article_summary"' in artifact_content
    assert '"judge"' in artifact_content


@pytest.mark.asyncio
async def test_pipeline_runner_records_crawl_exclusion_and_revision(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Story A")],
            "draft_outline": [_outline([urls[0]]), _outline([urls[0]])],
            "judge_evaluation": [
                {
                    "critique": "Tighten the heading.",
                    "specific_fixes_required": ["Shorten heading"],
                    "passes_criteria": False,
                },
                {
                    "critique": "Now it passes.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                },
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=urls[0], text="Article text A", metadata={}, title="Story A", origin_url=urls[0])],
            failures=[CrawlFailure(url=urls[1], reason="timeout")],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2))

    assert result.draft_result.revision_count == 1
    assert result.metadata.retries["draft_revision"] == 1
    assert result.summary_result.excluded[0].url == urls[1]
    assert result.summary_result.excluded[0].stage == "crawl"


@pytest.mark.asyncio
async def test_pipeline_runner_uses_newsletter_split_fallback(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://www.deeplearning.ai/the-batch/data-points"
    _write_input(tmp_path, target_date, [url])

    newsletter_text = """
In today's edition of Data Points

**First Item**
This is the first technical block with enough words to be treated as a real article segment and it includes engineering details for the report.
https://example.com/first

**Second Item**
This is the second technical block with enough words to also be treated as a real article segment and it includes enterprise governance detail for the report.
https://example.com/second
""".strip()

    llm_client = FakeLLMClient(
        structured={
            "newsletter_split": [StructuredOutputError("invalid markers")],
            "article_summary": [_summary("First Item"), _summary("Second Item")],
            "draft_outline": [_outline(["https://example.com/first", "https://example.com/second"])],
            "judge_evaluation": [
                {
                    "critique": "Pass.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=url, text=newsletter_text, metadata={}, title="Newsletter", origin_url=url)],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.summary_result.newsletter_splits
    assert result.summary_result.newsletter_splits[0].strategy == "heuristic_fallback"
    assert result.summary_result.newsletter_splits[0].artifact_paths
    assert any("invalid markers" in item for item in result.metadata.validation_failures)
    assert result.metadata.fallbacks == [f"newsletter_split:{url}"]


@pytest.mark.asyncio
async def test_pipeline_runner_exits_when_no_valid_summaries(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://example.com/a"
    _write_input(tmp_path, target_date, [url])

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [StructuredOutputError("missing summary fields")],
        },
        text=[],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=url, text="Article text", metadata={}, title="Story A", origin_url=url)],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )

    with pytest.raises(SystemExit, match="No valid articles after crawl/summarization. First errors: missing summary fields"):
        await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))


@pytest.mark.asyncio
async def test_pipeline_runner_keeps_input_order(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Story A"), _summary("Story B")],
            "draft_outline": [_outline(urls)],
            "judge_evaluation": [
                {
                    "critique": "Ready.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(url=urls[1], text="Article text B", metadata={}, title="Story B", origin_url=urls[1]),
                CrawlItem(url=urls[0], text="Article text A", metadata={}, title="Story A", origin_url=urls[0]),
            ],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2))

    assert [summary.url for summary in result.summary_result.summaries] == urls


@pytest.mark.asyncio
async def test_pipeline_runner_handles_partial_split_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target_date = date(2026, 3, 13)
    url = "https://example.com/news"
    _write_input(tmp_path, target_date, [url])

    split_items = [
        CrawlItem(url="https://example.com/news/part-1", text="Part 1", metadata={}, title="Part 1", origin_url=url),
        CrawlItem(url="https://example.com/news/part-2", text="Part 2", metadata={}, title="Part 2", origin_url=url),
    ]

    async def fake_split_articles(client, item, *, max_items=6, event_sink=None):
        del client, event_sink, max_items
        return NewsletterSplitResult(
            origin_url=item.url,
            items=split_items,
            strategy="heuristic_fallback",
        )

    monkeypatch.setattr("src.application.pipeline.split_newsletter_items_async", fake_split_articles)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Part 1"), StructuredOutputError("missing fields")],
            "draft_outline": [_outline([split_items[0].url])],
            "judge_evaluation": [
                {
                    "critique": "Acceptable.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=url,
                    text="Newsletter text",
                    metadata={"published": "2026-03-11"},
                    title="Newsletter",
                    origin_url=url,
                )
            ],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.summary_result.summaries[0].url == split_items[0].url
    assert result.summary_result.summaries[0].date == date(2026, 3, 11)
    assert not result.summary_result.excluded
    assert any(split_items[1].url in entry for entry in result.metadata.validation_failures)


@pytest.mark.asyncio
async def test_pipeline_runner_records_missing_crawl(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/exists", "https://example.com/missing"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Exists")],
            "draft_outline": [_outline([urls[0]])],
            "judge_evaluation": [
                {
                    "critique": "Solid.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=urls[0], text="Article text", metadata={}, title="Story", origin_url=urls[0])],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.summary_result.excluded[0].url == urls[1]
    assert result.summary_result.excluded[0].stage == "crawl"
    assert result.summary_result.excluded[0].reason == "crawl result missing"


@pytest.mark.asyncio
async def test_pipeline_runner_skips_split_artifacts_for_single_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target_date = date(2026, 3, 13)
    url = "https://example.com/news/single"
    _write_input(tmp_path, target_date, [url])

    async def fake_single_split(client, item, *, max_items=6, event_sink=None):
        del client, event_sink, max_items
        return NewsletterSplitResult(origin_url=item.url, items=[item], strategy="not_applicable")

    monkeypatch.setattr("src.application.pipeline.split_newsletter_items_async", fake_single_split)

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Single")],
            "draft_outline": [_outline([url])],
            "judge_evaluation": [
                {
                    "critique": "Fine.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=url, text="Article text", metadata={}, title="Short story", origin_url=url)],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.summary_result.newsletter_splits == []


@pytest.mark.asyncio
async def test_pipeline_runner_produces_debug_dir(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://example.com/debug"
    _write_input(tmp_path, target_date, [url])

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Debug")],
            "draft_outline": [_outline([url])],
            "judge_evaluation": [
                {
                    "critique": "Debugged.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=url, text="Article text", metadata={}, title="Debug story", origin_url=url)],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.persistence.debug_dir is not None
    assert result.persistence.debug_dir.exists()


@pytest.mark.asyncio
async def test_pipeline_runner_exposes_llm_usage_metrics(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://example.com/usage"
    _write_input(tmp_path, target_date, [url])

    llm_client = FakeLLMClient(
        structured={
            "article_summary": [_summary("Usage Story")],
            "draft_outline": [_outline([url])],
            "judge_evaluation": [
                {
                    "critique": "Looks good.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            ],
        },
        text=["## 1. Agentic Delivery\n\n### <u>**Execution speed improves**</u>\n"],
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[CrawlItem(url=url, text="Article text", metadata={}, title="Usage story", origin_url=url)],
            failures=[],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))

    assert result.metadata.llm_usage["totals"]["request_count"] == 4
    assert result.metadata.llm_usage["by_task"]["article_summary"]["input_tokens"] == 120
    assert result.metadata.llm_usage["by_role"]["writer"]["output_tokens"] == 105
