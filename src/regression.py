from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from .application.pipeline import PipelineRequest, run_report_pipeline
from .config import get_settings
from .domain.models import CrawlFailure, CrawlItem, CrawlStageResult
from .infrastructure.crawl_service import Crawl4AICrawlService
from .infrastructure.events import NullEventSink
from .infrastructure.llm_client import OpenAILLMClient
from .ingest import load_input
from .llm import OpenAIConfig
from .storage_paths import dated_input_path


@dataclass(slots=True)
class RunSnapshot:
    lane_id: str
    mode: str
    model: str
    family: str
    input_urls: list[str]
    evaluation_enabled: bool
    crawl_ok_count: int
    crawl_failures: list[dict[str, str]]
    story_cards: list[dict[str, object]]
    story_units: list[dict[str, object]]
    excluded: list[dict[str, object]]
    newsletter_splits: list[dict[str, object]]
    outline: dict[str, object]
    evaluation: dict[str, object]
    revision_count: int
    report_text: str
    report_sections: list[str]
    article_headings: list[str]
    artifact_path: Path | None
    report_path: Path | None
    error: str | None


@dataclass(slots=True)
class RegressionMatrixResult:
    output_dir: Path
    json_path: Path
    markdown_path: Path
    baseline: RunSnapshot
    lanes: list[dict[str, object]]
    verdicts: dict[str, bool]


class ReplayCrawlService:
    def __init__(
        self,
        *,
        items_by_url: dict[str, CrawlItem],
        failures_by_url: dict[str, str],
    ) -> None:
        self._items_by_url = items_by_url
        self._failures_by_url = failures_by_url

    @classmethod
    def from_paths(cls, artifact_path: str | Path, debug_dir: str | Path) -> ReplayCrawlService:
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        debug_path = Path(debug_dir)
        items_by_url = _load_replay_items(debug_path)
        failures_by_url = _extract_failure_map(payload)
        return cls(items_by_url=items_by_url, failures_by_url=failures_by_url)

    async def crawl(self, urls: list[str], *, max_concurrency: int) -> CrawlStageResult:
        del max_concurrency
        items: list[CrawlItem] = []
        failures: list[CrawlFailure] = []

        for url in urls:
            failure_reason = self._failures_by_url.get(url)
            if failure_reason:
                failures.append(CrawlFailure(url=url, reason=failure_reason))
                continue

            item = self._items_by_url.get(url)
            if item is None:
                failures.append(CrawlFailure(url=url, reason="replay input missing debug snapshot"))
                continue

            items.append(item)

        return CrawlStageResult(items=items, failures=failures)


async def run_regression_matrix(
    *,
    baseline_artifact_path: str | Path,
    baseline_report_path: str | Path,
    live_input_path: str | Path,
    output_dir: str | Path | None = None,
    max_concurrency: int = 3,
) -> RegressionMatrixResult:
    baseline_artifact = Path(baseline_artifact_path)
    baseline_report = Path(baseline_report_path)
    live_input = Path(live_input_path)

    bundle_dir = _resolve_output_dir(output_dir)
    lanes_root = bundle_dir / "lanes"
    lanes_root.mkdir(parents=True, exist_ok=True)

    baseline = load_run_snapshot(
        artifact_path=baseline_artifact,
        report_path=baseline_report,
        lane_id="historical_xai",
        mode="historical",
        model="xai",
        family="xai",
        error=None,
    )

    replay_target_date = _parse_date_from_path(baseline_report, prefix="", suffix="_weekly.md")
    live_target_date = _parse_date_from_path(live_input, prefix="links_", suffix=".yaml")
    live_payload = _load_input_payload(live_input)
    replay_debug_dir = (
        baseline_artifact.parent / "debug_inputs" / _run_id_from_artifact_path(baseline_artifact)
    )

    lanes: list[dict[str, object]] = []
    for mode in ("replay", "live"):
        lane_id = f"{mode}_routed_openai"
        lane_output_dir = lanes_root / lane_id
        lane_target_date = replay_target_date if mode == "replay" else live_target_date
        lane_urls = baseline.input_urls if mode == "replay" else list(live_payload["urls"])
        lane_evaluation = (
            baseline.evaluation_enabled if mode == "replay" else bool(live_payload["evaluation"])
        )
        lane_workspace, artifact_path, report_path, error = await _execute_lane(
            lane_id=lane_id,
            mode=mode,
            lane_output_dir=lane_output_dir,
            target_date=lane_target_date,
            urls=lane_urls,
            evaluation_enabled=lane_evaluation,
            max_concurrency=max_concurrency,
            baseline_artifact_path=baseline_artifact,
            replay_debug_dir=replay_debug_dir,
        )
        snapshot = _build_lane_snapshot(
            lane_id=lane_id,
            mode=mode,
            artifact_path=artifact_path,
            report_path=report_path,
            family="openai",
            input_urls=lane_urls,
            evaluation_enabled=lane_evaluation,
            error=error,
        )
        comparison = compare_snapshots(baseline, snapshot)
        lanes.append(
            {
                "lane_id": lane_id,
                "mode": mode,
                "model": snapshot.model,
                "family": snapshot.family,
                "workspace_dir": str(lane_workspace),
                "artifact_path": str(artifact_path) if artifact_path else None,
                "report_path": str(report_path) if report_path else None,
                "error": error,
                "snapshot": _snapshot_to_dict(snapshot),
                "comparison": comparison,
            }
        )

    verdicts = _build_overall_verdicts(lanes)
    json_path = bundle_dir / "regression_summary.json"
    markdown_path = bundle_dir / "regression_summary.md"

    payload = {
        "baseline": _snapshot_to_dict(baseline),
        "lanes": lanes,
        "verdicts": verdicts,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_summary(payload), encoding="utf-8")

    return RegressionMatrixResult(
        output_dir=bundle_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        baseline=baseline,
        lanes=lanes,
        verdicts=verdicts,
    )


def load_run_snapshot(
    *,
    artifact_path: str | Path,
    report_path: str | Path | None,
    lane_id: str,
    mode: str,
    model: str,
    family: str,
    error: str | None,
) -> RunSnapshot:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    input_payload = payload.get("input") or {}
    input_urls = [str(url) for url in input_payload.get("urls") or []]
    evaluation_enabled = bool(input_payload.get("evaluation_enabled", True))

    crawl_ok_count, crawl_failures = _extract_crawl_metrics(payload, input_urls)
    outline = _extract_outline(payload)
    evaluation = _extract_evaluation(payload)
    revision_count = _extract_revision_count(payload)
    report_text = ""
    report_file = Path(report_path) if report_path else None
    if report_file is not None and report_file.exists():
        report_text = report_file.read_text(encoding="utf-8")

    report_sections = _extract_report_sections(report_text)
    article_headings = _extract_article_headings(report_text, outline)
    story_cards = _extract_story_cards(payload)
    story_units = _extract_story_units(payload, outline)
    return RunSnapshot(
        lane_id=lane_id,
        mode=mode,
        model=model,
        family=family,
        input_urls=input_urls,
        evaluation_enabled=evaluation_enabled,
        crawl_ok_count=crawl_ok_count,
        crawl_failures=crawl_failures,
        story_cards=story_cards,
        story_units=story_units,
        excluded=[dict(item) for item in payload.get("excluded") or []],
        newsletter_splits=[dict(item) for item in payload.get("newsletter_splits") or []],
        outline=outline,
        evaluation=evaluation,
        revision_count=revision_count,
        report_text=report_text,
        report_sections=report_sections,
        article_headings=article_headings,
        artifact_path=Path(artifact_path),
        report_path=report_file,
        error=error,
    )


def compare_snapshots(baseline: RunSnapshot, candidate: RunSnapshot) -> dict[str, object]:
    baseline_input = set(baseline.input_urls)
    candidate_input = set(candidate.input_urls)
    baseline_failures = {item["url"]: item["reason"] for item in baseline.crawl_failures}
    candidate_failures = {item["url"]: item["reason"] for item in candidate.crawl_failures}
    baseline_cards = {str(item.get("url")): item for item in baseline.story_cards}
    candidate_cards = {str(item.get("url")): item for item in candidate.story_cards}
    baseline_excluded_stage_counts = dict(
        Counter(str(item.get("stage", "unknown")) for item in baseline.excluded)
    )
    candidate_excluded_stage_counts = dict(
        Counter(str(item.get("stage", "unknown")) for item in candidate.excluded)
    )

    title_changes = []
    date_changes = []
    confidence_deltas = []
    for url in sorted(baseline_cards.keys() & candidate_cards.keys()):
        baseline_item = baseline_cards[url]
        candidate_item = candidate_cards[url]
        baseline_title = str(
            baseline_item.get("story_title_tr") or baseline_item.get("title") or ""
        )
        candidate_title = str(
            candidate_item.get("story_title_tr") or candidate_item.get("title") or ""
        )
        if baseline_title != candidate_title:
            title_changes.append(
                {
                    "url": url,
                    "baseline_title": baseline_title,
                    "candidate_title": candidate_title,
                }
            )

        baseline_date = str(baseline_item.get("published_at") or baseline_item.get("date") or "")
        candidate_date = str(candidate_item.get("published_at") or candidate_item.get("date") or "")
        baseline_inferred = bool(
            baseline_item.get("published_at_inferred", baseline_item.get("date_inferred"))
        )
        candidate_inferred = bool(
            candidate_item.get("published_at_inferred", candidate_item.get("date_inferred"))
        )
        if baseline_date != candidate_date or baseline_inferred != candidate_inferred:
            date_changes.append(
                {
                    "url": url,
                    "baseline_date": baseline_date,
                    "candidate_date": candidate_date,
                    "baseline_date_inferred": baseline_inferred,
                    "candidate_date_inferred": candidate_inferred,
                }
            )

        baseline_confidence = float(baseline_item.get("confidence") or 0.0)
        candidate_confidence = float(candidate_item.get("confidence") or 0.0)
        if baseline_confidence != candidate_confidence:
            confidence_deltas.append(
                {
                    "url": url,
                    "baseline_confidence": baseline_confidence,
                    "candidate_confidence": candidate_confidence,
                    "delta": round(candidate_confidence - baseline_confidence, 3),
                }
            )

    baseline_themes = _extract_theme_names(baseline.outline)
    candidate_themes = _extract_theme_names(candidate.outline)
    candidate_outline_urls = _extract_outline_urls(candidate.outline)
    candidate_duplicate_outline_urls = _duplicate_urls(candidate_outline_urls)
    candidate_missing_outline_urls = sorted(set(candidate_cards) - set(candidate_outline_urls))
    missing_sections = sorted(set(baseline.report_sections) - set(candidate.report_sections))
    removed_headings = sorted(set(baseline.article_headings) - set(candidate.article_headings))
    added_headings = sorted(set(candidate.article_headings) - set(baseline.article_headings))
    candidate_eval_passes = bool(candidate.evaluation.get("passes_criteria", False))
    baseline_eval_passes = bool(baseline.evaluation.get("passes_criteria", False))

    regression_signals: list[str] = []
    if candidate.error:
        regression_signals.append("lane execution failed")
    if candidate.crawl_ok_count < baseline.crawl_ok_count:
        regression_signals.append("crawl ok count dropped")
    if len(candidate.crawl_failures) > len(baseline.crawl_failures):
        regression_signals.append("crawl failures increased")
    if len(candidate.story_cards) < len(baseline.story_cards):
        regression_signals.append("story card count dropped")
    if len(candidate.excluded) > len(baseline.excluded):
        regression_signals.append("excluded count increased")
    if candidate_eval_passes is False and baseline_eval_passes is True:
        regression_signals.append("evaluation no longer passes")
    if len(candidate_themes) < len(baseline_themes):
        regression_signals.append("theme count dropped")
    if len(candidate.story_units) < len(baseline.story_units):
        regression_signals.append("story unit count dropped")
    if candidate_duplicate_outline_urls:
        regression_signals.append("duplicate outline urls present")
    if candidate_missing_outline_urls:
        regression_signals.append("outline missing story card urls")
    if missing_sections:
        regression_signals.append("baseline report sections missing")
    if removed_headings:
        regression_signals.append("baseline article headings removed")

    return {
        "input": {
            "baseline_count": len(baseline.input_urls),
            "candidate_count": len(candidate.input_urls),
            "missing_urls": sorted(baseline_input - candidate_input),
            "extra_urls": sorted(candidate_input - baseline_input),
        },
        "crawl": {
            "baseline_ok_count": baseline.crawl_ok_count,
            "candidate_ok_count": candidate.crawl_ok_count,
            "baseline_failure_count": len(baseline.crawl_failures),
            "candidate_failure_count": len(candidate.crawl_failures),
            "failure_delta": len(candidate.crawl_failures) - len(baseline.crawl_failures),
            "new_failed_urls": sorted(set(candidate_failures) - set(baseline_failures)),
            "resolved_failed_urls": sorted(set(baseline_failures) - set(candidate_failures)),
            "failure_reason_changes": _failure_reason_changes(
                baseline_failures, candidate_failures
            ),
        },
        "story_cards": {
            "baseline_count": len(baseline.story_cards),
            "candidate_count": len(candidate.story_cards),
            "baseline_excluded_count": len(baseline.excluded),
            "candidate_excluded_count": len(candidate.excluded),
            "baseline_excluded_stage_counts": baseline_excluded_stage_counts,
            "candidate_excluded_stage_counts": candidate_excluded_stage_counts,
            "missing_story_card_urls": sorted(set(baseline_cards) - set(candidate_cards)),
            "extra_story_card_urls": sorted(set(candidate_cards) - set(baseline_cards)),
            "title_changes": title_changes,
            "date_changes": date_changes,
            "confidence_deltas": confidence_deltas,
        },
        "workflow": {
            "baseline_theme_count": len(baseline_themes),
            "candidate_theme_count": len(candidate_themes),
            "baseline_story_unit_count": len(baseline.story_units),
            "candidate_story_unit_count": len(candidate.story_units),
            "baseline_evaluation_passes": baseline_eval_passes,
            "candidate_evaluation_passes": candidate_eval_passes,
            "evaluation_pass_changed": baseline_eval_passes != candidate_eval_passes,
            "baseline_revision_count": baseline.revision_count,
            "candidate_revision_count": candidate.revision_count,
            "duplicate_outline_url_count": len(candidate_duplicate_outline_urls),
            "missing_outline_url_count": len(candidate_missing_outline_urls),
        },
        "report": {
            "baseline_length": len(baseline.report_text),
            "candidate_length": len(candidate.report_text),
            "baseline_heading_count": len(baseline.report_sections)
            + len(baseline.article_headings),
            "candidate_heading_count": len(candidate.report_sections)
            + len(candidate.article_headings),
            "missing_baseline_sections": missing_sections,
            "removed_article_headings": removed_headings,
            "added_article_headings": added_headings,
        },
        "verdict": {
            "regressed": bool(regression_signals),
            "signals": regression_signals,
        },
    }


async def _execute_lane(
    *,
    lane_id: str,
    mode: str,
    lane_output_dir: Path,
    target_date: date,
    urls: list[str],
    evaluation_enabled: bool,
    max_concurrency: int,
    baseline_artifact_path: Path,
    replay_debug_dir: Path,
) -> tuple[Path, Path | None, Path | None, str | None]:
    lane_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = dated_input_path(target_date, base_dir=lane_output_dir)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(_render_input_yaml(urls, evaluation_enabled), encoding="utf-8")

    settings = get_settings()
    llm_config = OpenAIConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_s=settings.openai_timeout_s,
        reasoning_effort=settings.openai_reasoning_effort,
        verbosity=settings.openai_verbosity,
        max_output_tokens=settings.openai_max_output_tokens,
        rpm_limit=settings.openai_rpm_limit,
        tpm_limit=settings.openai_tpm_limit,
        tpd_limit=settings.openai_tpd_limit,
    )
    llm_client = OpenAILLMClient(llm_config)
    crawl_service = (
        ReplayCrawlService.from_paths(baseline_artifact_path, replay_debug_dir)
        if mode == "replay"
        else Crawl4AICrawlService()
    )

    try:
        result = await run_report_pipeline(
            llm_client=llm_client,
            crawl_service=crawl_service,
            request=PipelineRequest(
                target_date=target_date,
                max_concurrency=max_concurrency,
                base_dir=lane_output_dir,
            ),
            event_sink=NullEventSink(),
        )
    except BaseException as exc:  # pragma: no cover - exercised in failure tests
        return lane_output_dir, None, None, str(exc)

    artifact_path = result.persistence.artifact_path
    report_path = result.persistence.report_path
    return lane_output_dir, artifact_path, report_path, None


def _build_lane_snapshot(
    *,
    lane_id: str,
    mode: str,
    artifact_path: Path | None,
    report_path: Path | None,
    family: str,
    input_urls: list[str],
    evaluation_enabled: bool,
    error: str | None,
) -> RunSnapshot:
    if artifact_path is None:
        return RunSnapshot(
            lane_id=lane_id,
            mode=mode,
            model="routed_openai",
            family=family,
            input_urls=input_urls,
            evaluation_enabled=evaluation_enabled,
            crawl_ok_count=0,
            crawl_failures=[],
            story_cards=[],
            story_units=[],
            excluded=[],
            newsletter_splits=[],
            outline={},
            evaluation={},
            revision_count=0,
            report_text="",
            report_sections=[],
            article_headings=[],
            artifact_path=None,
            report_path=report_path,
            error=error,
        )

    return load_run_snapshot(
        artifact_path=artifact_path,
        report_path=report_path,
        lane_id=lane_id,
        mode=mode,
        model="routed_openai",
        family=family,
        error=error,
    )


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        resolved_path = Path(output_dir)
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path
    return Path(tempfile.mkdtemp(prefix="ai-news-regression-"))


def _load_input_payload(path: Path) -> dict[str, object]:
    input_data = load_input(path)
    return {
        "urls": list(input_data.urls),
        "evaluation": bool(input_data.eval_enabled),
    }


def _extract_crawl_metrics(
    payload: dict[str, object], input_urls: list[str]
) -> tuple[int, list[dict[str, str]]]:
    crawl_payload = payload.get("crawl")
    if isinstance(crawl_payload, dict):
        ok_count = int(crawl_payload.get("ok_count") or 0)
        failures = [
            {"url": str(item.get("url") or ""), "reason": str(item.get("reason") or "")}
            for item in crawl_payload.get("failures") or []
        ]
        return ok_count, failures

    failures = [
        {"url": str(item.get("url") or ""), "reason": str(item.get("reason") or "")}
        for item in payload.get("crawl_failures") or []
    ]
    ok_count = max(0, len(input_urls) - len(failures))
    return ok_count, failures


def _extract_story_cards(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("story_cards"):
        return [dict(item) for item in payload.get("story_cards") or []]

    return [
        {
            "url": item.get("url"),
            "origin_url": item.get("origin_url"),
            "source_name": item.get("source_name"),
            "title_raw": item.get("title"),
            "published_at": item.get("date"),
            "published_at_inferred": item.get("date_inferred"),
            "story_title_tr": item.get("title"),
            "story_type": "legacy_summary",
            "key_facts": [item.get("summary_tr")] if item.get("summary_tr") else [],
            "must_keep_entities": [],
            "must_keep_facts": [item.get("summary_tr")] if item.get("summary_tr") else [],
            "why_it_matters_tr": item.get("why_it_matters_tr"),
            "technical_relevance": 0.0,
            "strategic_relevance": 0.0,
            "confidence": item.get("confidence", 0.0),
        }
        for item in payload.get("summaries") or []
    ]


def _extract_story_units(
    payload: dict[str, object], outline: dict[str, object]
) -> list[dict[str, object]]:
    if payload.get("story_units"):
        return [dict(item) for item in payload.get("story_units") or []]

    story_units: list[dict[str, object]] = []
    for theme in outline.get("themes") or []:
        if not isinstance(theme, dict):
            continue
        for article in theme.get("articles") or []:
            if not isinstance(article, dict):
                continue
            story_units.append(
                {
                    "story_unit_id": article.get("primary_url"),
                    "primary_url": article.get("primary_url"),
                    "supporting_url": None,
                    "news_urls_included": list(article.get("news_urls_included") or []),
                    "canonical_title": article.get("heading"),
                }
            )
    return story_units


def _extract_outline(payload: dict[str, object]) -> dict[str, object]:
    workflow_payload = payload.get("workflow")
    if isinstance(workflow_payload, dict):
        outline = workflow_payload.get("outline")
        if isinstance(outline, dict):
            return outline

    drafts_payload = payload.get("drafts")
    if isinstance(drafts_payload, dict):
        outline = drafts_payload.get("draft_outline")
        if isinstance(outline, dict):
            return outline

    return {}


def _extract_evaluation(payload: dict[str, object]) -> dict[str, object]:
    workflow_payload = payload.get("workflow")
    if isinstance(workflow_payload, dict):
        evaluation = workflow_payload.get("evaluation")
        if isinstance(evaluation, dict):
            return evaluation

    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        return evaluation

    return {}


def _extract_revision_count(payload: dict[str, object]) -> int:
    workflow_payload = payload.get("workflow")
    if isinstance(workflow_payload, dict):
        revision_count = workflow_payload.get("revision_count")
        if revision_count is not None:
            return int(revision_count)

    metadata_payload = payload.get("metadata")
    if isinstance(metadata_payload, dict):
        retries = metadata_payload.get("retries")
        if isinstance(retries, dict):
            return int(retries.get("draft_revision") or 0)

    return 0


def _extract_report_sections(report_text: str) -> list[str]:
    return [
        _clean_heading_text(line[3:]) for line in report_text.splitlines() if line.startswith("## ")
    ]


def _extract_article_headings(report_text: str, outline: dict[str, object]) -> list[str]:
    headings = [
        _clean_heading_text(line[4:])
        for line in report_text.splitlines()
        if line.startswith("### ")
    ]
    if headings:
        return headings

    outline_headings: list[str] = []
    for theme in outline.get("themes") or []:
        if not isinstance(theme, dict):
            continue
        for article in theme.get("articles") or []:
            if not isinstance(article, dict):
                continue
            heading = _clean_heading_text(str(article.get("heading") or ""))
            if heading:
                outline_headings.append(heading)
    return outline_headings


def _extract_theme_names(outline: dict[str, object]) -> list[str]:
    names: list[str] = []
    for theme in outline.get("themes") or []:
        if not isinstance(theme, dict):
            continue
        name = str(theme.get("theme_name") or "").strip()
        if name:
            names.append(name)
    return names


def _extract_outline_urls(outline: dict[str, object]) -> list[str]:
    urls: list[str] = []
    for theme in outline.get("themes") or []:
        if not isinstance(theme, dict):
            continue
        for article in theme.get("articles") or []:
            if not isinstance(article, dict):
                continue
            urls.extend(
                str(url).strip()
                for url in article.get("news_urls_included") or []
                if str(url).strip()
            )
    return urls


def _duplicate_urls(urls: list[str]) -> list[str]:
    counts = Counter(urls)
    return sorted(url for url, count in counts.items() if count > 1)


def _failure_reason_changes(
    baseline_failures: dict[str, str],
    candidate_failures: dict[str, str],
) -> list[dict[str, str]]:
    changed: list[dict[str, str]] = []
    for url in sorted(set(baseline_failures) & set(candidate_failures)):
        if baseline_failures[url] == candidate_failures[url]:
            continue
        changed.append(
            {
                "url": url,
                "baseline_reason": baseline_failures[url],
                "candidate_reason": candidate_failures[url],
            }
        )
    return changed


def _build_overall_verdicts(lanes: list[dict[str, object]]) -> dict[str, bool]:
    lookup = {lane["mode"]: lane for lane in lanes}
    replay_regressed = _lane_regressed(lookup.get("replay"))
    live_regressed = _lane_regressed(lookup.get("live"))
    likely_downstream = replay_regressed
    likely_crawl = live_regressed and not replay_regressed
    requires_manual_review = (replay_regressed and live_regressed) or not (
        likely_downstream or likely_crawl
    )
    return {
        "replay_regressed": replay_regressed,
        "live_regressed": live_regressed,
        "likely_crawl_regression": likely_crawl,
        "likely_downstream_regression": likely_downstream,
        "requires_manual_review": requires_manual_review,
    }


def _lane_regressed(lane: dict[str, object] | None) -> bool:
    if lane is None:
        return False
    comparison = lane.get("comparison") or {}
    verdict = comparison.get("verdict") or {}
    return bool(verdict.get("regressed"))


def _render_markdown_summary(payload: dict[str, object]) -> str:
    baseline = payload["baseline"]
    lanes = payload["lanes"]
    verdicts = payload["verdicts"]

    lines = [
        "# Regression Summary",
        "",
        "## Overall Verdicts",
        "",
    ]
    for key, value in verdicts.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Lane: `{baseline['lane_id']}`",
            f"- Model family: `{baseline['family']}`",
            f"- Input URLs: `{len(baseline['input_urls'])}`",
            f"- Story cards: `{len(baseline['story_cards'])}`",
            f"- Story units: `{len(baseline['story_units'])}`",
            f"- Themes: `{len(_extract_theme_names(baseline['outline']))}`",
        ]
    )

    for lane in lanes:
        comparison = lane["comparison"]
        lines.extend(
            [
                "",
                f"## Lane `{lane['lane_id']}`",
                "",
                f"- Mode: `{lane['mode']}`",
                f"- Model: `{lane['model']}`",
                f"- Regressed: `{comparison['verdict']['regressed']}`",
                f"- Signals: `{', '.join(comparison['verdict']['signals']) or 'none'}`",
                "- Story card count: "
                f"`{comparison['story_cards']['candidate_count']}` "
                f"vs baseline `{comparison['story_cards']['baseline_count']}`",
                "- Story unit count: "
                f"`{comparison['workflow']['candidate_story_unit_count']}` "
                f"vs baseline `{comparison['workflow']['baseline_story_unit_count']}`",
                "- Duplicate outline URLs: "
                f"`{comparison['workflow']['duplicate_outline_url_count']}`",
                "- Missing outline URLs: "
                f"`{comparison['workflow']['missing_outline_url_count']}`",
                "- Missing sections: "
                f"`{', '.join(comparison['report']['missing_baseline_sections']) or 'none'}`",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _snapshot_to_dict(snapshot: RunSnapshot) -> dict[str, object]:
    payload = asdict(snapshot)
    payload["artifact_path"] = str(snapshot.artifact_path) if snapshot.artifact_path else None
    payload["report_path"] = str(snapshot.report_path) if snapshot.report_path else None
    return payload


def _load_replay_items(debug_dir: Path) -> dict[str, CrawlItem]:
    items: dict[str, CrawlItem] = {}
    for path in sorted(debug_dir.glob("*.txt")):
        item = _parse_debug_input(path)
        if item is None:
            continue
        items[item.url] = item
    return items


def _parse_debug_input(path: Path) -> CrawlItem | None:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^Title:\s*(.*)$", text, flags=re.MULTILINE)
    url_match = re.search(r"^URL:\s*(.*)$", text, flags=re.MULTILINE)
    body = text.split("\n\n", 1)[1] if "\n\n" in text else ""
    url = (url_match.group(1).strip() if url_match else "").strip()
    if not url:
        return None
    title = (title_match.group(1).strip() if title_match else "").strip() or None
    return CrawlItem(
        url=url,
        text=body.strip(),
        metadata={},
        title=title,
        origin_url=url,
    )


def _extract_failure_map(payload: dict[str, object]) -> dict[str, str]:
    _, failures = _extract_crawl_metrics(
        payload, [str(url) for url in (payload.get("input") or {}).get("urls") or []]
    )
    return {item["url"]: item["reason"] for item in failures}


def _parse_date_from_path(path: Path, *, prefix: str, suffix: str) -> date:
    name = path.name
    stripped_name = name
    if prefix and stripped_name.startswith(prefix):
        stripped_name = stripped_name[len(prefix) :]
    if suffix and stripped_name.endswith(suffix):
        stripped_name = stripped_name[: -len(suffix)]
    direct_match = re.fullmatch(r"\d{2}-\d{2}-\d{4}", stripped_name)
    if direct_match:
        return datetime.strptime(stripped_name, "%d-%m-%Y").date()

    partial_match = re.search(r"(\d{2}-\d{2}-\d{4})", name)
    if partial_match:
        return datetime.strptime(partial_match.group(1), "%d-%m-%Y").date()

    return datetime.now().date()


def _run_id_from_artifact_path(path: Path) -> str:
    name = path.stem
    if name.startswith("run_"):
        return name[4:]
    return name


def _render_input_yaml(urls: list[str], evaluation_enabled: bool) -> str:
    lines = [f"evaluation: {'true' if evaluation_enabled else 'false'}", "urls:"]
    for url in urls:
        lines.append(f'  - "{url}"')
    return "\n".join(lines) + "\n"


def _clean_heading_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = text.replace("*", "")
    text = text.replace("+", "")
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
