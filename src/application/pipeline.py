from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

from ..domain.models import (
    CrawlItem,
    CrawlStageResult,
    ExcludedItem,
    InputLoadResult,
    NewsletterSplitResult,
    PipelineRunMetadata,
    PipelineRunResult,
    StoryCard,
    StoryCardStageResult,
    StorySetResult,
)
from ..infrastructure.events import (
    CompositeEventSink,
    ConsoleEventSink,
    EventSink,
    NullEventSink,
    PipelineEvent,
)
from ..infrastructure.llm_client import LLMClient, StructuredOutputError
from ..infrastructure.persistence import FileSystemPipelineStore
from ..ingest import load_input
from .content_tasks import (
    extract_story_card_async,
    prepare_crawl_item,
    split_newsletter_items_async,
)
from .report_tasks import build_story_set, classify_story_merges
from .report_workflow import ReportWorkflowService


@dataclass(slots=True)
class PipelineRequest:
    max_concurrency: int
    target_date: date = field(default_factory=lambda: datetime.now().date())
    base_dir: str | Path | None = None


@dataclass(slots=True)
class UrlProcessingResult:
    story_cards: list[StoryCard]
    excluded: list[ExcludedItem]
    newsletter_splits: list[NewsletterSplitResult]
    validation_failures: list[str]
    fallbacks: list[str]
    debug_paths: list[Path]


@dataclass(slots=True)
class StoryCardStageComputation:
    result: StoryCardStageResult
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
class StoryCardExecutionResult:
    story_cards: list[StoryCard]
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
        self.workflow = workflow or ReportWorkflowService(
            llm_client=llm_client, event_sink=event_sink
        )
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
            lambda: InputLoadResult(
                path=paths.input_path, data=self.input_loader(paths.input_path)
            ),
        )
        self._emit("LOAD_INPUT", f"path={input_result.path}")

        crawl_result = await self._time_stage_async(
            metadata,
            "crawl",
            lambda: self.crawl_service.crawl(
                input_result.data.urls, max_concurrency=request.max_concurrency
            ),
        )
        self._emit(
            "CRAWL",
            f"done ok={len(crawl_result.items)} failed={len(crawl_result.failures)}",
        )

        raw_dir = self.store.write_raw_texts(paths.output_path, crawl_result.items, metadata.run_id)
        self._emit("RAW_SAVE", f"saved={len(crawl_result.items)}")

        story_card_computation = await self._time_stage_async(
            metadata,
            "story_cards",
            lambda: self._extract_story_cards(
                urls=input_result.data.urls,
                crawl_result=crawl_result,
                output_path=paths.output_path,
                run_id=metadata.run_id,
            ),
        )
        story_card_result = story_card_computation.result
        metadata.validation_failures.extend(
            self._dedupe(story_card_computation.validation_failures)
        )
        metadata.fallbacks.extend(self._dedupe(story_card_computation.fallbacks))

        if not story_card_result.story_cards:
            failure_message = self._build_no_valid_articles_message(
                story_card_result=story_card_result,
                validation_failures=story_card_computation.validation_failures,
            )
            raise SystemExit(failure_message)

        self._emit(
            "STORY_CARDS",
            "included="
            f"{len(story_card_result.story_cards)} "
            "excluded="
            f"{len(story_card_result.excluded)}",
        )

        story_set_result = await self._time_stage_async(
            metadata,
            "story_set",
            lambda: self._build_story_set(story_card_result.story_cards),
        )
        self._emit(
            "STORY_SET",
            f"units={len(story_set_result.story_units)} "
            f"pairs={len(story_set_result.candidate_pairs)}",
        )

        draft_result = await self._time_stage_async(
            metadata,
            "workflow",
            lambda: self.workflow.run(
                story_units=story_set_result.story_units,
                excluded=story_card_result.excluded,
                eval_enabled=input_result.data.eval_enabled,
            ),
        )
        metadata.retries["draft_revision"] = draft_result.revision_count
        metadata.llm_usage = self._collect_llm_usage()

        persistence = self._time_stage(
            metadata,
            "persist",
            lambda: self.store.persist_run(
                paths=paths,
                input_result=input_result,
                crawl_result=crawl_result,
                story_card_result=story_card_result,
                story_set_result=story_set_result,
                draft_result=draft_result,
                metadata=metadata,
                raw_dir=raw_dir,
                debug_dir=story_card_computation.debug_dir,
            ),
        )
        self._emit(
            "WRITE_OUTPUT",
            f"report={persistence.report_path} artifacts={persistence.artifact_path}",
        )
        return PipelineRunResult(
            paths=paths,
            input_result=input_result,
            crawl_result=crawl_result,
            story_card_result=story_card_result,
            story_set_result=story_set_result,
            draft_result=draft_result,
            persistence=persistence,
            metadata=metadata,
        )

    async def _extract_story_cards(
        self,
        *,
        urls: list[str],
        crawl_result: CrawlStageResult,
        output_path: Path,
        run_id: str,
    ) -> StoryCardStageComputation:
        crawl_map, failure_map = self._build_crawl_maps(crawl_result)
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
        return self._aggregate_story_card_results(results)

    async def _build_story_set(self, story_cards: list[StoryCard]) -> StorySetResult:
        merge_decisions = await classify_story_merges(self.llm_client, story_cards)
        return build_story_set(story_cards, [], merge_decisions)

    def _build_crawl_maps(
        self, crawl_result: CrawlStageResult
    ) -> tuple[dict[str, CrawlItem], dict[str, str]]:
        crawl_map = {item.url: item for item in crawl_result.items}
        failure_map = {item.url: item.reason for item in crawl_result.failures}
        return crawl_map, failure_map

    def _aggregate_story_card_results(
        self, results: list[UrlProcessingResult]
    ) -> StoryCardStageComputation:
        story_cards: list[StoryCard] = []
        excluded: list[ExcludedItem] = []
        newsletter_splits: list[NewsletterSplitResult] = []
        validation_failures: list[str] = []
        fallbacks: list[str] = []
        debug_paths: list[Path] = []

        for result in results:
            story_cards.extend(result.story_cards)
            excluded.extend(result.excluded)
            newsletter_splits.extend(result.newsletter_splits)
            validation_failures.extend(result.validation_failures)
            fallbacks.extend(result.fallbacks)
            debug_paths.extend(result.debug_paths)

        debug_dir = debug_paths[0].parent if debug_paths else None
        source_texts = {story_card.url: story_card.raw_text for story_card in story_cards}
        return StoryCardStageComputation(
            result=StoryCardStageResult(
                story_cards=story_cards,
                excluded=excluded,
                newsletter_splits=newsletter_splits,
                source_texts=source_texts,
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

        prepared_split = await self._prepare_split_result(
            item=item,
            output_path=output_path,
            run_id=run_id,
        )
        execution = await self._extract_story_cards_for_split_items(
            items=prepared_split.items,
            output_path=output_path,
            run_id=run_id,
        )

        if execution.story_cards:
            return UrlProcessingResult(
                story_cards=execution.story_cards,
                excluded=[],
                newsletter_splits=prepared_split.newsletter_splits,
                validation_failures=prepared_split.validation_failures
                + execution.validation_failures,
                fallbacks=prepared_split.fallbacks,
                debug_paths=execution.debug_paths,
            )

        return self._excluded_result(
            url=url,
            reason=execution.errors[0] if execution.errors else "story card extraction failed",
            stage="story_card",
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
            split_result.items = [
                self._inherit_parent_metadata(item, split_item) for split_item in split_result.items
            ]
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

    async def _extract_story_cards_for_split_items(
        self,
        *,
        items: list[CrawlItem],
        output_path: Path,
        run_id: str,
    ) -> StoryCardExecutionResult:
        debug_paths: list[Path] = []
        story_cards: list[StoryCard] = []
        validation_failures: list[str] = []
        errors: list[str] = []

        for item in items:
            try:
                story_card = await self._extract_story_card(
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

            story_cards.append(story_card)

        return StoryCardExecutionResult(
            story_cards=story_cards,
            validation_failures=validation_failures,
            errors=errors,
            debug_paths=debug_paths,
        )

    async def _extract_story_card(
        self,
        *,
        item: CrawlItem,
        output_path: Path,
        run_id: str,
        debug_paths: list[Path],
    ) -> StoryCard:
        debug_paths.append(self.store.write_debug_input(output_path, run_id, item))
        return await extract_story_card_async(self.llm_client, item)

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
            story_cards=[],
            excluded=[ExcludedItem(url=url, reason=reason, stage=stage)],
            newsletter_splits=newsletter_splits or [],
            validation_failures=validation_failures or [],
            fallbacks=fallbacks or [],
            debug_paths=debug_paths or [],
        )

    def _emit(self, stage: str, message: str = "") -> None:
        self.event_sink.emit(PipelineEvent(stage=stage, message=message))

    def _inherit_parent_metadata(self, parent_item: CrawlItem, split_item: CrawlItem) -> CrawlItem:
        merged_metadata = dict(parent_item.metadata)
        merged_metadata.update(split_item.metadata)
        inherited_item = CrawlItem(
            url=split_item.url,
            text=split_item.text,
            metadata=merged_metadata,
            title=split_item.title or parent_item.title,
            origin_url=split_item.origin_url or parent_item.origin_url or parent_item.url,
            title_raw=split_item.title_raw
            or parent_item.title_raw
            or split_item.title
            or parent_item.title,
            source_name=split_item.source_name or parent_item.source_name,
            source_family=split_item.source_family or parent_item.source_family,
            published_at=split_item.published_at or parent_item.published_at,
            published_at_inferred=split_item.published_at_inferred
            if split_item.published_at is not None
            else parent_item.published_at_inferred,
            content_type=split_item.content_type or parent_item.content_type,
            crawl_quality_flags=list(
                split_item.crawl_quality_flags or parent_item.crawl_quality_flags
            ),
            blocked_or_partial=split_item.blocked_or_partial or parent_item.blocked_or_partial,
        )
        return prepare_crawl_item(inherited_item)

    def _collect_llm_usage(self) -> dict[str, object]:
        get_usage_summary = getattr(self.llm_client, "get_usage_summary", None)
        if not callable(get_usage_summary):
            return {}

        usage_summary = get_usage_summary()
        if isinstance(usage_summary, dict):
            return usage_summary
        return {}

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

    def _build_no_valid_articles_message(
        self,
        *,
        story_card_result: StoryCardStageResult,
        validation_failures: list[str],
    ) -> str:
        base_message = "No valid articles after crawl/story-card extraction."
        reason_candidates: list[str] = []

        for failure in validation_failures:
            normalized_reason = self._strip_failure_prefix(failure)
            if normalized_reason:
                reason_candidates.append(normalized_reason)

        for excluded_item in story_card_result.excluded:
            if excluded_item.reason:
                reason_candidates.append(excluded_item.reason)

        unique_reasons = self._dedupe(reason_candidates)
        if not unique_reasons:
            return base_message

        preview_reasons = unique_reasons[:3]
        preview_text = "; ".join(preview_reasons)
        return f"{base_message} First errors: {preview_text}"

    def _strip_failure_prefix(self, failure: str) -> str:
        if ": " not in failure:
            return failure
        _, reason = failure.split(": ", 1)
        return reason


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
