from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from src.application.pipeline import PipelineRequest, PipelineRunner
from src.domain.models import CrawlFailure, CrawlItem, CrawlStageResult, NewsletterSplitResult
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
    def __init__(self, *, structured: dict[str, list] | None = None) -> None:
        self.structured = {key: list(value) for key, value in (structured or {}).items()}
        self.task_call_counts: dict[str, int] = {}
        self.usage_snapshot = {
            "totals": {
                "request_count": 6,
                "input_tokens": 830,
                "output_tokens": 310,
                "cached_input_tokens": 90,
                "reasoning_tokens": 18,
                "total_tokens": 1140,
            },
            "by_task": {
                "story_card_extraction": {
                    "request_count": 2,
                    "input_tokens": 200,
                    "output_tokens": 90,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 290,
                },
                "theme_assignment": {
                    "request_count": 1,
                    "input_tokens": 210,
                    "output_tokens": 70,
                    "cached_input_tokens": 15,
                    "reasoning_tokens": 8,
                    "total_tokens": 280,
                },
                "intro_writer": {
                    "request_count": 1,
                    "input_tokens": 70,
                    "output_tokens": 25,
                    "cached_input_tokens": 10,
                    "reasoning_tokens": 3,
                    "total_tokens": 95,
                },
                "story_article": {
                    "request_count": 1,
                    "input_tokens": 180,
                    "output_tokens": 65,
                    "cached_input_tokens": 25,
                    "reasoning_tokens": 4,
                    "total_tokens": 245,
                },
                "cod_gelisme": {
                    "request_count": 1,
                    "input_tokens": 170,
                    "output_tokens": 60,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 3,
                    "total_tokens": 230,
                },
            },
            "by_role": {
                "researcher": {
                    "request_count": 2,
                    "input_tokens": 200,
                    "output_tokens": 90,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 290,
                },
                "planner": {
                    "request_count": 1,
                    "input_tokens": 210,
                    "output_tokens": 70,
                    "cached_input_tokens": 15,
                    "reasoning_tokens": 8,
                    "total_tokens": 280,
                },
                "writer": {
                    "request_count": 3,
                    "input_tokens": 420,
                    "output_tokens": 150,
                    "cached_input_tokens": 55,
                    "reasoning_tokens": 10,
                    "total_tokens": 570,
                },
            },
        }

    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        del system
        self.task_call_counts[task_name] = self.task_call_counts.get(task_name, 0) + 1
        queue = self.structured.get(task_name)
        if queue:
            value = queue.pop(0)
            if isinstance(value, Exception):
                raise value
            return schema.model_validate(value)

        if task_name == "theme_assignment":
            story_unit_ids = []
            for story_unit_id in re.findall(r'"story_unit_id":\s*"([^"]+)"', user):
                if story_unit_id not in story_unit_ids:
                    story_unit_ids.append(story_unit_id)
            return schema.model_validate(
                {
                    "report_title": "Weekly AI Report",
                    "introduction_signal": "Signals",
                    "themes": [
                        {
                            "theme_name": "1. Agentic Delivery",
                            "theme_commentary": "Theme overview.",
                            "story_unit_ids": story_unit_ids,
                        }
                    ],
                }
            )

        if task_name == "judge_evaluation":
            return schema.model_validate(
                {
                    "critique": "Looks good.",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                }
            )

        if task_name == "repair_planner":
            return schema.model_validate({"critique": "No repairs required.", "operations": []})

        if task_name == "intro_writer":
            return schema.model_validate(
                {
                    "introduction_commentary": (
                        "Bu hafta agentic delivery odakli urun ve "
                        "workflow sinyalleri öne çıktı."
                    )
                }
            )

        if task_name == "story_article":
            return schema.model_validate(
                {
                    "gelisme": (
                        "OpenAI ve Anthropic gibi ekipler agentic "
                        "delivery odakli net urun sinyalleri verdi."
                    ),
                    "neden_onemli": (
                        "Bu degisim kurumsal gelistirme akisini daha "
                        "uygulanabilir hale getiriyor."
                    ),
                }
            )

        if task_name == "cod_gelisme":
            required_additions = re.findall(
                r"Required additions for this round:\n(.*)\n\nCurrent Gelişme:",
                user,
                flags=re.DOTALL,
            )
            additions = required_additions[0].strip() if required_additions else "OpenAI"
            return schema.model_validate(
                {
                    "missing_entities": [],
                    "gelisme": (
                        f"{additions} odakli duyuru, agentic delivery "
                        "hattinda platform orkestrasyonu, gelistirici "
                        "workflow uyumu, entegrasyon netligi, uygulama "
                        "ici kontrol, operasyonel gorunurluk ve daha "
                        "uygulanabilir kurumsal kullanim senaryolari "
                        "etrafinda toplanan somut bir urun sinyali sundu; "
                        "degisiklik, ekiplerin yeni yetenekleri daha az "
                        "belirsizlikle production benzeri akislara "
                        "tasimasini kolaylastiran daha olgun bir uygulama "
                        "zemini sagladi."
                    ),
                }
            )

        if task_name == "merge_classifier":
            return schema.model_validate({"merges": []})

        raise AssertionError(f"unexpected structured task: {task_name}")

    async def generate_text(self, *, system: str, user: str) -> str:  # pragma: no cover - unused
        del system, user
        raise AssertionError("unexpected text generation call")

    def get_usage_summary(self):
        return self.usage_snapshot


def _write_input(
    base_dir: Path, target_date: date, urls: list[str], evaluation: bool = True
) -> Path:
    input_path = dated_input_path(target_date, base_dir=base_dir)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "evaluation: "
        + ("true" if evaluation else "false")
        + "\n"
        + "urls:\n"
        + "".join(f'  - "{url}"\n' for url in urls),
        encoding="utf-8",
    )
    return input_path


def _story_card_payload(
    title: str, *, entities: list[str] | None = None, facts: list[str] | None = None
) -> dict:
    return {
        "story_title_tr": title,
        "story_type": "product_update",
        "key_facts": facts or [f"{title} fact"],
        "must_keep_entities": entities or [title.split()[0]],
        "must_keep_facts": facts or [f"{title} fact"],
        "why_it_matters_tr": f"{title} matters for delivery.",
        "technical_relevance": 0.75,
        "strategic_relevance": 0.8,
        "confidence": 0.85,
    }


@pytest.mark.asyncio
async def test_pipeline_runner_persists_story_card_run(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload("Story A", entities=["OpenAI"], facts=["OpenAI released A"]),
                _story_card_payload(
                    "Story B", entities=["Anthropic"], facts=["Anthropic released B"]
                ),
            ]
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=urls[0],
                    text="Article text A",
                    metadata={},
                    title="Story A",
                    origin_url=urls[0],
                ),
                CrawlItem(
                    url=urls[1],
                    text="Article text B",
                    metadata={},
                    title="Story B",
                    origin_url=urls[1],
                ),
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
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2)
    )

    assert result.persistence.report_path.exists()
    assert result.persistence.artifact_path.exists()
    artifact_content = result.persistence.artifact_path.read_text(encoding="utf-8")
    assert '"story_cards"' in artifact_content
    assert '"story_units"' in artifact_content
    assert '"theme_plan"' in artifact_content
    assert '"story_card_extraction"' in artifact_content
    assert result.persistence.debug_dir is not None
    assert len(result.story_card_result.story_cards) == 2
    assert len(result.story_set_result.story_units) == 2


@pytest.mark.asyncio
async def test_pipeline_runner_fails_open_when_merge_classifier_errors(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload(
                    "OpenAI launches Agents SDK",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
                _story_card_payload(
                    "Agents SDK launch expands OpenAI platform",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
            ],
            "merge_classifier": [StructuredOutputError("merge classifier failed")],
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=urls[0],
                    text="Article text A",
                    metadata={},
                    title="Story A",
                    origin_url=urls[0],
                ),
                CrawlItem(
                    url=urls[1],
                    text="Article text B",
                    metadata={},
                    title="Story B",
                    origin_url=urls[1],
                ),
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

    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2)
    )

    assert len(result.story_set_result.story_units) == 2
    assert {
        story_unit.primary_url for story_unit in result.story_set_result.story_units
    } == set(urls)


@pytest.mark.asyncio
async def test_pipeline_runner_uses_single_merge_classifier_request(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
        "https://example.com/d",
    ]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload(
                    "OpenAI launches Agents SDK",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
                _story_card_payload(
                    "Agents SDK launch expands OpenAI platform",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
                _story_card_payload(
                    "OpenAI ships another Agents SDK update",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
                _story_card_payload(
                    "OpenAI platform update references Agents SDK",
                    entities=["OpenAI", "Agents SDK"],
                    facts=["OpenAI launched Agents SDK"],
                ),
            ]
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=url,
                    text=f"Article text for {url}",
                    metadata={},
                    title=url,
                    origin_url=url,
                )
                for url in urls
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

    await runner.run(PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=4))

    assert llm_client.task_call_counts["merge_classifier"] == 1


@pytest.mark.asyncio
async def test_pipeline_runner_records_crawl_exclusion(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/a", "https://example.com/b"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload("Story A", entities=["OpenAI"], facts=["OpenAI released A"])
            ]
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=urls[0],
                    text="Article text A",
                    metadata={},
                    title="Story A",
                    origin_url=urls[0],
                )
            ],
            failures=[CrawlFailure(url=urls[1], reason="timeout")],
        )
    )

    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=FileSystemPipelineStore(),
        event_sink=NullEventSink(),
    )
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=2)
    )

    assert result.story_card_result.excluded[0].url == urls[1]
    assert result.story_card_result.excluded[0].stage == "crawl"
    assert result.story_card_result.excluded[0].reason == "timeout"


@pytest.mark.asyncio
async def test_pipeline_runner_uses_newsletter_split_fallback(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://www.deeplearning.ai/the-batch/data-points"
    _write_input(tmp_path, target_date, [url])

    newsletter_text = """
In today's edition of Data Points

**First Item**
This is the first technical block with enough words to be treated as a real
article segment and it includes engineering details for the report.
https://example.com/first

**Second Item**
This is the second technical block with enough words to also be treated as a
real article segment and it includes enterprise governance detail for the report.
https://example.com/second
""".strip()

    llm_client = FakeLLMClient(
        structured={
            "newsletter_split": [StructuredOutputError("invalid markers")],
            "story_card_extraction": [
                _story_card_payload("First Item", entities=["OpenAI"], facts=["First item fact"]),
                _story_card_payload(
                    "Second Item", entities=["Anthropic"], facts=["Second item fact"]
                ),
            ],
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=url, text=newsletter_text, metadata={}, title="Newsletter", origin_url=url
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
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1)
    )

    assert result.story_card_result.newsletter_splits
    assert result.story_card_result.newsletter_splits[0].strategy == "heuristic_fallback"
    assert result.story_card_result.newsletter_splits[0].artifact_paths


@pytest.mark.asyncio
async def test_pipeline_runner_handles_partial_split_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target_date = date(2026, 3, 13)
    url = "https://example.com/newsletter"
    _write_input(tmp_path, target_date, [url])

    split_items = [
        CrawlItem(
            url="https://example.com/news/part-1",
            text="Part 1",
            metadata={},
            title="Part 1",
            origin_url=url,
        ),
        CrawlItem(
            url="https://example.com/news/part-2",
            text="Part 2",
            metadata={},
            title="Part 2",
            origin_url=url,
        ),
    ]

    async def fake_split_articles(client, item, *, max_items=6, event_sink=None):
        del client, item, max_items, event_sink
        return NewsletterSplitResult(
            origin_url=url, items=split_items, strategy="heuristic_fallback"
        )

    monkeypatch.setattr(
        "src.application.pipeline.split_newsletter_items_async", fake_split_articles
    )

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload("Part 1", entities=["OpenAI"], facts=["Part 1 fact"]),
                StructuredOutputError("missing fields"),
            ]
        }
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
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1)
    )

    assert result.story_card_result.story_cards[0].url == split_items[0].url
    assert result.story_card_result.story_cards[0].published_at == date(2026, 3, 11)
    assert not result.story_card_result.excluded


@pytest.mark.asyncio
async def test_pipeline_runner_records_missing_crawl_result(tmp_path: Path):
    target_date = date(2026, 3, 13)
    urls = ["https://example.com/exists", "https://example.com/missing"]
    _write_input(tmp_path, target_date, urls)

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload("Exists", entities=["OpenAI"], facts=["Exists fact"])
            ]
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=urls[0], text="Article text", metadata={}, title="Story", origin_url=urls[0]
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
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1)
    )

    assert result.story_card_result.excluded[0].url == urls[1]
    assert result.story_card_result.excluded[0].stage == "crawl"
    assert result.story_card_result.excluded[0].reason == "crawl result missing"


@pytest.mark.asyncio
async def test_pipeline_runner_exposes_llm_usage_metrics(tmp_path: Path):
    target_date = date(2026, 3, 13)
    url = "https://example.com/usage"
    _write_input(tmp_path, target_date, [url])

    llm_client = FakeLLMClient(
        structured={
            "story_card_extraction": [
                _story_card_payload("Usage Story", entities=["OpenAI"], facts=["Usage fact"])
            ]
        }
    )
    crawl_service = FakeCrawlService(
        CrawlStageResult(
            items=[
                CrawlItem(
                    url=url, text="Article text", metadata={}, title="Usage story", origin_url=url
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
    result = await runner.run(
        PipelineRequest(target_date=target_date, base_dir=tmp_path, max_concurrency=1)
    )

    assert result.metadata.llm_usage["by_task"]["story_card_extraction"]["input_tokens"] == 200
    assert result.metadata.llm_usage["by_role"]["researcher"]["request_count"] == 2
