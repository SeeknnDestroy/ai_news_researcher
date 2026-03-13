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
from .content_tasks import split_newsletter_items_async, summarize_article_async
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


@dataclass(slots=True)
class PreparedSplitResult:
    items: list[CrawlItem]
    newsletter_splits: list[NewsletterSplitResult]
    validation_failures: list[str]
    fallbacks: list[str]


@dataclass(slots=True)
class SummaryExecutionResult:
    summaries: list[SummaryItem]
    validation_failures: list[str]
    errors: list[str]
    debug_paths: list[Path]


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
        crawl_map, failure_map = self._build_crawl_maps(crawl_result)
        tasks = [self._process_url(url=url, crawl_map=crawl_map, failure_map=failure_map, output_path=output_path, run_id=run_id) for url in urls]
        results = await asyncio.gather(*tasks)
        return self._aggregate_summary_results(results)

    def _build_crawl_maps(self, crawl_result: CrawlStageResult) -> tuple[dict[str, CrawlItem], dict[str, str]]:
        crawl_map = {item.url: item for item in crawl_result.items}
        failure_map = {item.url: item.reason for item in crawl_result.failures}
        return crawl_map, failure_map

    def _aggregate_summary_results(self, results: list[UrlProcessingResult]) -> SummaryStageComputation:
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
            return self._excluded_result(url=url, reason=failure_map[url], stage="crawl")

        item = crawl_map.get(url)
        if item is None:
            return self._excluded_result(url=url, reason="crawl result missing", stage="crawl")

        date_result = extract_date(item.metadata, item.text, item.url)
        prepared_split = await self._prepare_split_result(
            item=item,
            output_path=output_path,
            run_id=run_id,
        )
        execution = await self._summarize_split_items(
            items=prepared_split.items,
            output_path=output_path,
            run_id=run_id,
            date_value=date_result.value,
            date_inferred=date_result.inferred,
        )

        if execution.summaries:
            return UrlProcessingResult(
                summaries=execution.summaries,
                excluded=[],
                newsletter_splits=prepared_split.newsletter_splits,
                validation_failures=prepared_split.validation_failures + execution.validation_failures,
                fallbacks=prepared_split.fallbacks,
                debug_paths=execution.debug_paths,
            )

        return self._excluded_result(
            url=url,
            reason=execution.errors[0] if execution.errors else "summary generation failed",
            stage="summarize",
            newsletter_splits=prepared_split.newsletter_splits,
            validation_failures=prepared_split.validation_failures + execution.validation_failures,
            fallbacks=prepared_split.fallbacks,
            debug_paths=execution.debug_paths,
        )

    async def _prepare_split_result(
        self,
        *,
        item: CrawlItem,
        output_path: Path,
        run_id: str,
    ) -> PreparedSplitResult:
        split_result = await split_newsletter_items_async(
            self.llm_client,
            item,
            event_sink=self.event_sink,
        )
        validation_failures: list[str] = []
        fallbacks: list[str] = []
        newsletter_splits: list[NewsletterSplitResult] = []

        if split_result.validation_error:
            validation_failures.append(f"{item.url}: {split_result.validation_error}")
        if split_result.strategy == "heuristic_fallback":
            fallbacks.append(f"newsletter_split:{item.url}")
        if len(split_result.items) > 1:
            split_result.artifact_paths = self.store.write_split_items(
                output_path,
                run_id,
                item.url,
                split_result.items,
            )
            newsletter_splits.append(split_result)
            self._emit("SPLIT_SAVE", f"origin={item.url} files={len(split_result.artifact_paths)}")

        return PreparedSplitResult(
            items=split_result.items,
            newsletter_splits=newsletter_splits,
            validation_failures=validation_failures,
            fallbacks=fallbacks,
        )

    async def _summarize_split_items(
        self,
        *,
        items: list[CrawlItem],
        output_path: Path,
        run_id: str,
        date_value,
        date_inferred: bool,
    ) -> SummaryExecutionResult:
        debug_paths: list[Path] = []
        summaries: list[SummaryItem] = []
        validation_failures: list[str] = []
        errors: list[str] = []

        for item in items:
            try:
                summary = await self._summarize_item(
                    item=item,
                    output_path=output_path,
                    run_id=run_id,
                    debug_paths=debug_paths,
                )
            except StructuredOutputError as exc:
                validation_failures.append(f"{item.url}: {exc}")
                errors.append(str(exc))
                continue
            except Exception as exc:
                errors.append(str(exc))
                continue

            summary.date = date_value
            summary.date_inferred = date_inferred
            summaries.append(summary)

        return SummaryExecutionResult(
            summaries=summaries,
            validation_failures=validation_failures,
            errors=errors,
            debug_paths=debug_paths,
        )

    async def _summarize_item(
        self,
        *,
        item: CrawlItem,
        output_path: Path,
        run_id: str,
        debug_paths: list[Path],
    ) -> SummaryItem:
        debug_paths.append(self.store.write_debug_input(output_path, run_id, item))
        return await summarize_article_async(self.llm_client, item)

    def _excluded_result(
        self,
        *,
        url: str,
        reason: str,
        stage: str,
        newsletter_splits: list[NewsletterSplitResult] | None = None,
        validation_failures: list[str] | None = None,
        fallbacks: list[str] | None = None,
        debug_paths: list[Path] | None = None,
    ) -> UrlProcessingResult:
        return UrlProcessingResult(
            summaries=[],
            excluded=[ExcludedItem(url=url, reason=reason, stage=stage)],
            newsletter_splits=newsletter_splits or [],
            validation_failures=validation_failures or [],
            fallbacks=fallbacks or [],
            debug_paths=debug_paths or [],
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
