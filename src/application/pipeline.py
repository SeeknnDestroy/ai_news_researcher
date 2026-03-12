from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

from ..date_extract import extract_date
from ..domain.models import (
    CrawlItem,
    CrawlStageResult,
    ExcludedItem,
    InputLoadResult,
    NewsletterSplitResult,
    PipelineRunMetadata,
    PipelineRunResult,
    SummaryItem,
    SummaryStageResult,
)
from ..ingest import load_input
from ..infrastructure.events import CompositeEventSink, ConsoleEventSink, EventSink, NullEventSink, PipelineEvent
from ..infrastructure.llm_client import LLMClient, StructuredOutputError
from ..infrastructure.persistence import FileSystemPipelineStore
from .ai_tasks import split_newsletter_items_async, summarize_article_async
from .report_workflow import ReportWorkflowService


@dataclass(slots=True)
class PipelineRequest:
    max_concurrency: int
    target_date: date = field(default_factory=lambda: datetime.now().date())
    base_dir: str | Path | None = None


@dataclass(slots=True)
class UrlProcessingResult:
    summaries: list[SummaryItem]
    excluded: list[ExcludedItem]
    newsletter_splits: list[NewsletterSplitResult]
    validation_failures: list[str]
    fallbacks: list[str]
    debug_paths: list[Path]


@dataclass(slots=True)
class SummaryStageComputation:
    result: SummaryStageResult
    validation_failures: list[str]
    fallbacks: list[str]
    debug_dir: Path | None


class PipelineRunner:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        crawl_service,
        store: FileSystemPipelineStore | None = None,
        workflow: ReportWorkflowService | None = None,
        event_sink: EventSink | None = None,
        input_loader=load_input,
    ) -> None:
        self.llm_client = llm_client
        self.crawl_service = crawl_service
        self.store = store or FileSystemPipelineStore()
        self.workflow = workflow or ReportWorkflowService(llm_client=llm_client, event_sink=event_sink)
        self.event_sink = event_sink or NullEventSink()
        self.input_loader = input_loader

    async def run(self, request: PipelineRequest) -> PipelineRunResult:
        started_at = datetime.now()
        metadata = PipelineRunMetadata(
            run_id=started_at.strftime("%d-%m-%Y_%H%M%S"),
            started_at=started_at,
        )
        paths = self.store.resolve_paths(target_date=request.target_date, base_dir=request.base_dir)

        input_result = self._time_stage(
            metadata,
            "load_input",
            lambda: InputLoadResult(path=paths.input_path, data=self.input_loader(paths.input_path)),
        )
        self._emit("LOAD_INPUT", f"path={input_result.path}")

        crawl_result = await self._time_stage_async(
            metadata,
            "crawl",
            lambda: self.crawl_service.crawl(input_result.data.urls, max_concurrency=request.max_concurrency),
        )
        self._emit(
            "CRAWL",
            f"done ok={len(crawl_result.items)} failed={len(crawl_result.failures)}",
        )

        raw_dir = self.store.write_raw_texts(paths.output_path, crawl_result.items, metadata.run_id)
        self._emit("RAW_SAVE", f"saved={len(crawl_result.items)}")

        summary_computation = await self._time_stage_async(
            metadata,
            "summarize",
            lambda: self._summarize_urls(
                urls=input_result.data.urls,
                crawl_result=crawl_result,
                output_path=paths.output_path,
                run_id=metadata.run_id,
            ),
        )
        summary_stage_result = summary_computation.result
        metadata.validation_failures.extend(self._dedupe(summary_computation.validation_failures))
        metadata.fallbacks.extend(self._dedupe(summary_computation.fallbacks))

        if not summary_stage_result.summaries:
            raise SystemExit("No valid articles after crawl/summarization.")

        self._emit(
            "FILTER",
            f"included={len(summary_stage_result.summaries)} excluded={len(summary_stage_result.excluded)}",
        )

        draft_result = await self._time_stage_async(
            metadata,
            "workflow",
            lambda: self.workflow.run(
                summaries=summary_stage_result.summaries,
                excluded=summary_stage_result.excluded,
                eval_enabled=input_result.data.eval_enabled,
            ),
        )
        metadata.retries["draft_revision"] = draft_result.revision_count

        persistence = self._time_stage(
            metadata,
            "persist",
            lambda: self.store.persist_run(
                paths=paths,
                input_result=input_result,
                crawl_result=crawl_result,
                summary_result=summary_stage_result,
                draft_result=draft_result,
                metadata=metadata,
                raw_dir=raw_dir,
                debug_dir=summary_computation.debug_dir,
            ),
        )
        self._emit("WRITE_OUTPUT", f"report={persistence.report_path} artifacts={persistence.artifact_path}")
        return PipelineRunResult(
            paths=paths,
            input_result=input_result,
            crawl_result=crawl_result,
            summary_result=summary_stage_result,
            draft_result=draft_result,
            persistence=persistence,
            metadata=metadata,
        )

    async def _summarize_urls(
        self,
        *,
        urls: list[str],
        crawl_result: CrawlStageResult,
        output_path: Path,
        run_id: str,
    ) -> SummaryStageComputation:
        crawl_map = {item.url: item for item in crawl_result.items}
        failure_map = {item.url: item.reason for item in crawl_result.failures}

        tasks = [
            self._process_url(
                url=url,
                crawl_map=crawl_map,
                failure_map=failure_map,
                output_path=output_path,
                run_id=run_id,
            )
            for url in urls
        ]
        results = await asyncio.gather(*tasks)

        summaries: list[SummaryItem] = []
        excluded: list[ExcludedItem] = []
        newsletter_splits: list[NewsletterSplitResult] = []
        validation_failures: list[str] = []
        fallbacks: list[str] = []
        debug_paths: list[Path] = []
        for result in results:
            summaries.extend(result.summaries)
            excluded.extend(result.excluded)
            newsletter_splits.extend(result.newsletter_splits)
            validation_failures.extend(result.validation_failures)
            fallbacks.extend(result.fallbacks)
            debug_paths.extend(result.debug_paths)

        debug_dir = debug_paths[0].parent if debug_paths else None
        return SummaryStageComputation(
            result=SummaryStageResult(
                summaries=summaries,
                excluded=excluded,
                newsletter_splits=newsletter_splits,
            ),
            validation_failures=validation_failures,
            fallbacks=fallbacks,
            debug_dir=debug_dir,
        )

    async def _process_url(
        self,
        *,
        url: str,
        crawl_map: dict[str, CrawlItem],
        failure_map: dict[str, str],
        output_path: Path,
        run_id: str,
    ) -> UrlProcessingResult:
        if url in failure_map:
            return UrlProcessingResult(
                summaries=[],
                excluded=[ExcludedItem(url=url, reason=failure_map[url], stage="crawl")],
                newsletter_splits=[],
                validation_failures=[],
                fallbacks=[],
                debug_paths=[],
            )

        item = crawl_map.get(url)
        if item is None:
            return UrlProcessingResult(
                summaries=[],
                excluded=[ExcludedItem(url=url, reason="crawl result missing", stage="crawl")],
                newsletter_splits=[],
                validation_failures=[],
                fallbacks=[],
                debug_paths=[],
            )

        date_result = extract_date(item.metadata, item.text, item.url)
        split_result = await split_newsletter_items_async(
            self.llm_client,
            item,
            event_sink=self.event_sink,
        )
        newsletter_splits: list[NewsletterSplitResult] = []
        validation_failures: list[str] = []
        fallbacks: list[str] = []

        if split_result.validation_error:
            validation_failures.append(f"{url}: {split_result.validation_error}")
        if split_result.strategy == "heuristic_fallback":
            fallbacks.append(f"newsletter_split:{url}")
        if len(split_result.items) > 1:
            split_result.artifact_paths = self.store.write_split_items(
                output_path,
                run_id,
                url,
                split_result.items,
            )
            newsletter_splits.append(split_result)
            self._emit("SPLIT_SAVE", f"origin={url} files={len(split_result.artifact_paths)}")

        debug_paths: list[Path] = []
        summaries: list[SummaryItem] = []
        errors: list[str] = []

        async def _summarize(item_to_process: CrawlItem) -> SummaryItem | None:
            debug_paths.append(self.store.write_debug_input(output_path, run_id, item_to_process))
            return await summarize_article_async(self.llm_client, item_to_process)

        for derived in split_result.items:
            try:
                summary = await _summarize(derived)
            except StructuredOutputError as exc:
                validation_failures.append(f"{derived.url}: {exc}")
                errors.append(str(exc))
                continue
            except Exception as exc:
                errors.append(str(exc))
                continue

            summary.date = date_result.value
            summary.date_inferred = date_result.inferred
            summaries.append(summary)

        if summaries:
            return UrlProcessingResult(
                summaries=summaries,
                excluded=[],
                newsletter_splits=newsletter_splits,
                validation_failures=validation_failures,
                fallbacks=fallbacks,
                debug_paths=debug_paths,
            )

        reason = errors[0] if errors else "summary generation failed"
        return UrlProcessingResult(
            summaries=[],
            excluded=[ExcludedItem(url=url, reason=reason, stage="summarize")],
            newsletter_splits=newsletter_splits,
            validation_failures=validation_failures,
            fallbacks=fallbacks,
            debug_paths=debug_paths,
        )

    def _emit(self, stage: str, message: str = "") -> None:
        self.event_sink.emit(PipelineEvent(stage=stage, message=message))

    def _time_stage(self, metadata: PipelineRunMetadata, stage_name: str, func):
        started = perf_counter()
        result = func()
        metadata.stage_timings[stage_name] = round(perf_counter() - started, 3)
        return result

    async def _time_stage_async(self, metadata: PipelineRunMetadata, stage_name: str, func):
        started = perf_counter()
        result = await func()
        metadata.stage_timings[stage_name] = round(perf_counter() - started, 3)
        return result

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result


async def run_report_pipeline(
    *,
    llm_client: LLMClient,
    crawl_service,
    request: PipelineRequest,
    store: FileSystemPipelineStore | None = None,
    event_sink: EventSink | None = None,
    workflow: ReportWorkflowService | None = None,
) -> PipelineRunResult:
    sink = event_sink or CompositeEventSink(ConsoleEventSink())
    runner = PipelineRunner(
        llm_client=llm_client,
        crawl_service=crawl_service,
        store=store,
        workflow=workflow,
        event_sink=sink,
    )
    return await runner.run(request)
