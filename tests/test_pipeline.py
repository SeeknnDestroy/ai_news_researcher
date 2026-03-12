from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from src.application.pipeline import PipelineRequest, PipelineRunner
from src.domain.models import CrawlFailure, CrawlItem, CrawlStageResult
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

    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        del system, user
        try:
            value = self.structured[task_name].pop(0)
        except (KeyError, IndexError) as exc:
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

    with pytest.raises(SystemExit, match="No valid articles after crawl/summarization."):
        await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1))
