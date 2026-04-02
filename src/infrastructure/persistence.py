from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..domain.contracts import DraftOutline, JudgeEvaluation, ThemeAssignmentPlan
from ..domain.models import (
    CrawlItem,
    CrawlStageResult,
    DraftWorkflowResult,
    InputLoadResult,
    PersistenceResult,
    PipelinePaths,
    PipelineRunMetadata,
    StoryCard,
    StoryCardStageResult,
    StorySetResult,
    StoryUnit,
)
from ..storage_paths import artifacts_root_for_output, dated_report_path, resolve_input_path
from ..utils import format_date, slugify_url


class PersistenceError(RuntimeError):
    pass


class FileSystemPipelineStore:
    def resolve_paths(self, *, target_date, base_dir: str | Path | None = None) -> PipelinePaths:
        input_path = resolve_input_path(target_date, base_dir=base_dir)
        output_path = dated_report_path(target_date, base_dir=base_dir)
        artifact_root = artifacts_root_for_output(output_path)
        return PipelinePaths(
            input_path=input_path, output_path=output_path, artifact_root=artifact_root
        )

    def write_raw_texts(
        self, out_path: str | Path, crawl_items: Iterable[CrawlItem], run_id: str
    ) -> Path:
        raw_dir = artifacts_root_for_output(out_path) / "raw" / run_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        for item in crawl_items:
            slug = slugify_url(item.url)
            file_path = raw_dir / f"{slug}.txt"
            file_path.write_text(item.text or "", encoding="utf-8")

        return raw_dir

    def write_split_items(
        self,
        out_path: str | Path,
        run_id: str,
        origin_url: str,
        items: list[CrawlItem],
    ) -> list[str]:
        base_dir = artifacts_root_for_output(out_path) / "splits" / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        origin_slug = slugify_url(origin_url)
        split_dir = base_dir / origin_slug
        split_dir.mkdir(parents=True, exist_ok=True)

        concat_path = split_dir / "_all_items_concat.txt"
        concat_content = [f"Source: {origin_url}\nTotal Items: {len(items)}\n{'=' * 50}\n"]

        paths: list[str] = []
        for idx, item in enumerate(items, start=1):
            path = split_dir / f"item_{idx:02d}.txt"
            item_text = (
                f"Item {idx}\n"
                f"Title: {item.title or ''}\n"
                f"URL: {item.url}\n"
                f"{'-' * 20}\n"
                f"{item.text or ''}\n"
            )
            path.write_text(item_text, encoding="utf-8")
            paths.append(str(path))
            concat_content.append(item_text)
            concat_content.append(f"\n{'=' * 50}\n")

        concat_path.write_text("\n".join(concat_content), encoding="utf-8")
        return paths

    def write_debug_input(self, out_path: str | Path, run_id: str, item: CrawlItem) -> Path:
        debug_dir = artifacts_root_for_output(out_path) / "debug_inputs" / run_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify_url(item.url)
        path = debug_dir / f"{slug}.txt"
        content = f"Title: {item.title}\nURL: {item.url}\n\n{item.text}"
        path.write_text(content, encoding="utf-8")
        return path

    def write_report(self, out_path: str | Path, report: str) -> Path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        return path

    def write_artifacts(
        self,
        *,
        paths: PipelinePaths,
        input_result: InputLoadResult,
        crawl_result: CrawlStageResult,
        story_card_result: StoryCardStageResult,
        story_set_result: StorySetResult,
        draft_result: DraftWorkflowResult,
        metadata: PipelineRunMetadata,
        raw_dir: Path | None,
    ) -> Path:
        artifact_path = paths.artifact_root / f"run_{metadata.run_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._artifact_payload(
            input_result=input_result,
            crawl_result=crawl_result,
            story_card_result=story_card_result,
            story_set_result=story_set_result,
            draft_result=draft_result,
            metadata=metadata,
            raw_dir=raw_dir,
        )
        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return artifact_path

    def persist_run(
        self,
        *,
        paths: PipelinePaths,
        input_result: InputLoadResult,
        crawl_result: CrawlStageResult,
        story_card_result: StoryCardStageResult,
        story_set_result: StorySetResult,
        draft_result: DraftWorkflowResult,
        metadata: PipelineRunMetadata,
        raw_dir: Path | None = None,
        debug_dir: Path | None = None,
    ) -> PersistenceResult:
        report_path = self.write_report(paths.output_path, draft_result.final_report)
        artifact_path = self.write_artifacts(
            paths=paths,
            input_result=input_result,
            crawl_result=crawl_result,
            story_card_result=story_card_result,
            story_set_result=story_set_result,
            draft_result=draft_result,
            metadata=metadata,
            raw_dir=raw_dir,
        )
        return PersistenceResult(
            report_path=report_path,
            artifact_path=artifact_path,
            raw_dir=raw_dir,
            debug_dir=debug_dir,
        )

    def _artifact_payload(
        self,
        *,
        input_result: InputLoadResult,
        crawl_result: CrawlStageResult,
        story_card_result: StoryCardStageResult,
        story_set_result: StorySetResult,
        draft_result: DraftWorkflowResult,
        metadata: PipelineRunMetadata,
        raw_dir: Path | None,
    ) -> dict:
        return {
            "input": {
                "path": str(input_result.path),
                "urls": input_result.data.urls,
                "evaluation_enabled": input_result.data.eval_enabled,
            },
            "crawl": {
                "ok_count": len(crawl_result.items),
                "failure_count": len(crawl_result.failures),
                "failures": [
                    {"url": item.url, "reason": item.reason} for item in crawl_result.failures
                ],
            },
            "story_cards": [
                self._serialize_story_card(item, raw_dir=raw_dir)
                for item in story_card_result.story_cards
            ],
            "story_units": [
                self._serialize_story_unit(item) for item in story_set_result.story_units
            ],
            "candidate_pairs": [
                {
                    "left_url": item.left_url,
                    "right_url": item.right_url,
                    "reason_codes": item.reason_codes,
                }
                for item in story_set_result.candidate_pairs
            ],
            "merge_decisions": [
                {
                    "left_url": item.left_url,
                    "right_url": item.right_url,
                    "decision": item.decision,
                    "rationale": item.rationale,
                }
                for item in story_set_result.merge_decisions
            ],
            "excluded": [
                {"url": item.url, "reason": item.reason, "stage": item.stage}
                for item in story_card_result.excluded
            ],
            "newsletter_splits": [
                {
                    "origin_url": split.origin_url,
                    "strategy": split.strategy,
                    "validation_error": split.validation_error,
                    "paths": split.artifact_paths,
                    "item_count": len(split.items),
                }
                for split in story_card_result.newsletter_splits
            ],
            "workflow": {
                "theme_plan": self._serialize_theme_plan(draft_result.theme_plan),
                "outline": self._serialize_outline(draft_result.outline),
                "evaluation": self._serialize_evaluation(draft_result.evaluation),
                "critique": draft_result.critique,
                "revision_count": draft_result.revision_count,
                "critique_history": draft_result.critique_history,
            },
            "metadata": {
                "run_id": metadata.run_id,
                "started_at": metadata.started_at.isoformat(),
                "stage_timings": metadata.stage_timings,
                "validation_failures": metadata.validation_failures,
                "retries": metadata.retries,
                "fallbacks": metadata.fallbacks,
                "llm_usage": metadata.llm_usage,
            },
        }

    def _serialize_story_card(self, item: StoryCard, *, raw_dir: Path | None) -> dict:
        raw_text_path = None
        if raw_dir is not None:
            raw_text_path = raw_dir / f"{slugify_url(item.url)}.txt"
        return {
            "url": item.url,
            "origin_url": item.origin_url,
            "source_name": item.source_name,
            "title_raw": item.title_raw,
            "published_at": format_date(item.published_at),
            "published_at_inferred": item.published_at_inferred,
            "content_type": item.content_type,
            "crawl_quality_flags": item.crawl_quality_flags,
            "blocked_or_partial": item.blocked_or_partial,
            "source_family": item.source_family,
            "story_title_tr": item.story_title_tr,
            "story_type": item.story_type,
            "key_facts": item.key_facts,
            "must_keep_entities": item.must_keep_entities,
            "must_keep_facts": item.must_keep_facts,
            "why_it_matters_tr": item.why_it_matters_tr,
            "technical_relevance": item.technical_relevance,
            "strategic_relevance": item.strategic_relevance,
            "confidence": item.confidence,
            "raw_text_path": str(raw_text_path) if raw_text_path is not None else None,
        }

    def _serialize_story_unit(self, item: StoryUnit) -> dict:
        return {
            "story_unit_id": item.story_unit_id,
            "primary_url": item.primary_url,
            "supporting_url": item.supporting_url,
            "merge_relation": item.merge_relation,
            "news_urls_included": item.news_urls_included,
            "canonical_title": item.canonical_title,
            "canonical_story_type": item.canonical_story_type,
            "key_facts": item.key_facts,
            "must_keep_entities": item.must_keep_entities,
            "must_keep_facts": item.must_keep_facts,
            "why_it_matters_tr": item.why_it_matters_tr,
            "technical_relevance": item.technical_relevance,
            "strategic_relevance": item.strategic_relevance,
            "confidence": item.confidence,
        }

    def _serialize_theme_plan(self, theme_plan: ThemeAssignmentPlan | None) -> dict | None:
        if theme_plan is None:
            return None
        return theme_plan.model_dump(mode="json")

    def _serialize_outline(self, outline: DraftOutline) -> dict:
        return outline.model_dump(mode="json")

    def _serialize_evaluation(self, evaluation: JudgeEvaluation) -> dict:
        return evaluation.model_dump(mode="json")
