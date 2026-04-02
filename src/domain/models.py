from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import DraftOutline, JudgeEvaluation, ThemeAssignmentPlan


@dataclass(slots=True)
class InputData:
    urls: list[str]
    eval_enabled: bool = True


@dataclass(slots=True)
class CrawlItem:
    url: str
    text: str
    metadata: dict[str, object]
    title: str | None = None
    origin_url: str | None = None
    title_raw: str | None = None
    source_name: str | None = None
    source_family: str | None = None
    published_at: date | None = None
    published_at_inferred: bool = True
    content_type: str = "text/html"
    crawl_quality_flags: list[str] = field(default_factory=list)
    blocked_or_partial: bool = False


@dataclass(slots=True)
class SummaryItem:
    url: str
    origin_url: str
    source_name: str
    title: str
    date: date | None
    date_inferred: bool
    summary_tr: str
    why_it_matters_tr: str
    tags: list[str]
    confidence: float


@dataclass(slots=True)
class ExcludedItem:
    url: str
    reason: str
    stage: str = "unknown"


@dataclass(slots=True)
class CrawlFailure:
    url: str
    reason: str


@dataclass(slots=True)
class NewsletterSplitResult:
    origin_url: str
    items: list[CrawlItem]
    strategy: str
    validation_error: str | None = None
    artifact_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InputLoadResult:
    path: Path
    data: InputData


@dataclass(slots=True)
class CrawlStageResult:
    items: list[CrawlItem]
    failures: list[CrawlFailure]


@dataclass(slots=True)
class SummaryStageResult:
    summaries: list[SummaryItem]
    excluded: list[ExcludedItem]
    newsletter_splits: list[NewsletterSplitResult]
    source_texts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StoryCard:
    url: str
    origin_url: str
    source_name: str
    title_raw: str
    published_at: date | None
    published_at_inferred: bool
    raw_text: str
    content_type: str
    crawl_quality_flags: list[str]
    blocked_or_partial: bool
    source_family: str
    story_title_tr: str
    story_type: str
    key_facts: list[str]
    must_keep_entities: list[str]
    must_keep_facts: list[str]
    why_it_matters_tr: str
    technical_relevance: float
    strategic_relevance: float
    confidence: float = 0.0


@dataclass(slots=True)
class StoryCardStageResult:
    story_cards: list[StoryCard]
    excluded: list[ExcludedItem]
    newsletter_splits: list[NewsletterSplitResult]
    source_texts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CandidatePair:
    left_url: str
    right_url: str
    reason_codes: list[str]


@dataclass(slots=True)
class MergeDecision:
    left_url: str
    right_url: str
    decision: str
    rationale: str = ""


@dataclass(slots=True)
class StoryUnit:
    story_unit_id: str
    story_cards: list[StoryCard]
    primary_url: str
    supporting_url: str | None
    merge_relation: str
    canonical_title: str
    canonical_story_type: str
    key_facts: list[str]
    must_keep_entities: list[str]
    must_keep_facts: list[str]
    why_it_matters_tr: str
    technical_relevance: float
    strategic_relevance: float
    confidence: float

    @property
    def news_urls_included(self) -> list[str]:
        urls = [card.url for card in self.story_cards]
        return sorted(urls, key=lambda value: (value != self.primary_url, value))

    @classmethod
    def from_story_cards(
        cls,
        *,
        story_cards: list[StoryCard],
        primary_url: str,
        merge_relation: str,
    ) -> StoryUnit:
        ordered_cards = sorted(story_cards, key=lambda card: card.url)
        joined_urls = "|".join(card.url for card in ordered_cards)
        story_unit_hash = hashlib.sha1(joined_urls.encode("utf-8")).hexdigest()[:12]
        story_unit_id = f"story-{story_unit_hash}"
        primary_card = next(card for card in ordered_cards if card.url == primary_url)
        supporting_card = next((card for card in ordered_cards if card.url != primary_url), None)
        key_facts = _merge_unique_strings(
            primary_card.key_facts,
            *(card.key_facts for card in ordered_cards if card.url != primary_url),
        )
        must_keep_entities = _merge_unique_strings(
            primary_card.must_keep_entities,
            *(card.must_keep_entities for card in ordered_cards if card.url != primary_url),
        )
        must_keep_facts = _merge_unique_strings(
            primary_card.must_keep_facts,
            *(card.must_keep_facts for card in ordered_cards if card.url != primary_url),
        )
        why_it_matters = primary_card.why_it_matters_tr or (
            supporting_card.why_it_matters_tr if supporting_card else ""
        )
        technical_relevance = max(card.technical_relevance for card in ordered_cards)
        strategic_relevance = max(card.strategic_relevance for card in ordered_cards)
        confidence = max(card.confidence for card in ordered_cards)
        return cls(
            story_unit_id=story_unit_id,
            story_cards=ordered_cards,
            primary_url=primary_url,
            supporting_url=supporting_card.url if supporting_card is not None else None,
            merge_relation=merge_relation,
            canonical_title=primary_card.story_title_tr,
            canonical_story_type=primary_card.story_type,
            key_facts=key_facts,
            must_keep_entities=must_keep_entities,
            must_keep_facts=must_keep_facts,
            why_it_matters_tr=why_it_matters,
            technical_relevance=technical_relevance,
            strategic_relevance=strategic_relevance,
            confidence=confidence,
        )


@dataclass(slots=True)
class StorySetResult:
    story_units: list[StoryUnit]
    candidate_pairs: list[CandidatePair]
    merge_decisions: list[MergeDecision]


@dataclass(slots=True)
class OutlineValidationResult:
    errors: list[str]
    failed_story_unit_ids: list[str] = field(default_factory=list)
    failed_theme_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DraftWorkflowResult:
    outline: DraftOutline
    theme_plan: ThemeAssignmentPlan | None
    evaluation: JudgeEvaluation
    final_report: str
    critique: str
    revision_count: int
    critique_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PersistenceResult:
    report_path: Path
    artifact_path: Path
    raw_dir: Path | None = None
    debug_dir: Path | None = None


@dataclass(slots=True)
class PipelinePaths:
    input_path: Path
    output_path: Path
    artifact_root: Path


@dataclass(slots=True)
class PipelineRunMetadata:
    run_id: str
    started_at: datetime
    stage_timings: dict[str, float] = field(default_factory=dict)
    validation_failures: list[str] = field(default_factory=list)
    retries: dict[str, int] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    llm_usage: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineRunResult:
    paths: PipelinePaths
    input_result: InputLoadResult
    crawl_result: CrawlStageResult
    story_card_result: StoryCardStageResult
    story_set_result: StorySetResult
    draft_result: DraftWorkflowResult
    persistence: PersistenceResult
    metadata: PipelineRunMetadata


def _merge_unique_strings(*value_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for values in value_groups:
        for value in values:
            normalized_value = str(value).strip()
            if not normalized_value:
                continue
            normalized_key = normalized_value.casefold()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            merged.append(normalized_value)
    return merged
