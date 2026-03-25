from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .contracts import DraftOutline, JudgeEvaluation


@dataclass(slots=True)
class InputData:
    urls: List[str]
    eval_enabled: bool = True


@dataclass(slots=True)
class CrawlItem:
    url: str
    text: str
    metadata: Dict[str, object]
    title: Optional[str] = None
    origin_url: Optional[str] = None


@dataclass(slots=True)
class SummaryItem:
    url: str
    origin_url: str
    source_name: str
    title: str
    date: Optional[date]
    date_inferred: bool
    summary_tr: str
    why_it_matters_tr: str
    tags: List[str]
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
    items: List[CrawlItem]
    strategy: str
    validation_error: Optional[str] = None
    artifact_paths: List[str] = field(default_factory=list)


@dataclass(slots=True)
class InputLoadResult:
    path: Path
    data: InputData


@dataclass(slots=True)
class CrawlStageResult:
    items: List[CrawlItem]
    failures: List[CrawlFailure]


@dataclass(slots=True)
class SummaryStageResult:
    summaries: List[SummaryItem]
    excluded: List[ExcludedItem]
    newsletter_splits: List[NewsletterSplitResult]
    source_texts: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DraftWorkflowResult:
    outline: "DraftOutline"
    evaluation: "JudgeEvaluation"
    final_report: str
    critique: str
    revision_count: int
    critique_history: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PersistenceResult:
    report_path: Path
    artifact_path: Path
    raw_dir: Optional[Path] = None
    debug_dir: Optional[Path] = None


@dataclass(slots=True)
class PipelinePaths:
    input_path: Path
    output_path: Path
    artifact_root: Path


@dataclass(slots=True)
class PipelineRunMetadata:
    run_id: str
    started_at: datetime
    stage_timings: Dict[str, float] = field(default_factory=dict)
    validation_failures: List[str] = field(default_factory=list)
    retries: Dict[str, int] = field(default_factory=dict)
    fallbacks: List[str] = field(default_factory=list)
    llm_usage: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineRunResult:
    paths: PipelinePaths
    input_result: InputLoadResult
    crawl_result: CrawlStageResult
    summary_result: SummaryStageResult
    draft_result: DraftWorkflowResult
    persistence: PersistenceResult
    metadata: PipelineRunMetadata
