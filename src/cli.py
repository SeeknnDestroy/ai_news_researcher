from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .config import CrawlItem, ExcludedItem, SummaryItem
from .crawler import CrawlError, crawl_urls
from .date_extract import extract_date
from .ingest import InputError, load_input
from .llm import XAIConfig
from .summarize import summarize_article
from .synthesize import synthesize_report
from .themes import group_themes
from .newsletter import split_newsletter_items
from .validate import (
    deterministic_checks,
    eval_result_to_dict,
    evaluate_report,
    find_link_urls,
    needs_revision,
    revise_report,
)
from .utils import format_date, slugify_url, log_progress, log_stage
from .drafts import write_draft, write_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="GenAI weekly report generator (MVP1)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate a weekly report from a URL list")
    run.add_argument("--model", default="grok-4-1-fast-reasoning", help="xAI model name")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--max-concurrency", type=int, default=3)

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)


def run_pipeline(args: argparse.Namespace) -> None:
    today = datetime.now().date()
    run_id = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    date_slug = today.strftime("%d-%m-%Y")
    input_path = Path("inputs") / f"links_{date_slug}.yaml"
    output_path = Path("reports") / f"{date_slug}_weekly.md"

    try:
        input_data = load_input(input_path)
    except InputError as exc:
        raise SystemExit(str(exc))

    log_stage(
        "LOAD_INPUT",
        f"path={input_path}",
    )

    llm_config = XAIConfig(
        model=args.model,
        temperature=args.temperature,
    )

    log_stage("CRAWL", f"urls={len(input_data.urls)}")
    try:
        crawl_items, crawl_failures = crawl_urls(
            input_data.urls, max_concurrency=args.max_concurrency
        )
    except CrawlError as exc:
        raise SystemExit(str(exc))
    log_stage("CRAWL", f"done ok={len(crawl_items)} failed={len(crawl_failures)}")

    _write_raw_text(str(output_path), crawl_items, run_id)
    log_stage("RAW_SAVE", f"saved={len(crawl_items)}")

    crawl_map = {item.url: item for item in crawl_items}
    failure_map = {url: reason for url, reason in crawl_failures}
    excluded: List[ExcludedItem] = []
    summaries: List[SummaryItem] = []
    source_texts = {}
    newsletter_splits: List[dict] = []

    total_urls = len(input_data.urls)
    for idx, url in enumerate(input_data.urls, start=1):
        log_progress("SUMMARY", idx, total_urls, url)
        if url in failure_map:
            reason = failure_map.get(url, "crawl failed")
            excluded.append(ExcludedItem(url=url, reason=reason))
            continue

        item = crawl_map.get(url)
        if not item:
            excluded.append(ExcludedItem(url=url, reason="crawl result missing"))
            continue

        date_result = extract_date(item.metadata, item.text, item.url)

        derived_items = split_newsletter_items(llm_config, item, max_items=6)
        if len(derived_items) > 1:
            log_stage("SPLIT_NEWSLETTER", f"{url} -> {len(derived_items)} items")
            split_paths = _write_split_items(str(output_path), run_id, url, derived_items)
            newsletter_splits.append({"origin_url": url, "paths": split_paths})
            if split_paths:
                log_stage("SPLIT_SAVE", f"origin={url} dir={Path(split_paths[0]).parent} files={len(split_paths)}")

        item_summaries = []
        for sub_index, derived in enumerate(derived_items, start=1):
            log_progress("SUMMARY", sub_index, len(derived_items), derived.url)
            
            # Debug: Save the text LLM will see
            _write_debug_input(str(output_path), run_id, derived)

            summary = summarize_article(llm_config, derived)
            summary.date = date_result.value
            summary.date_inferred = date_result.inferred

            if not summary.summary_tr or not summary.why_it_matters_tr:
                continue

            summaries.append(summary)
            item_summaries.append(summary)
            source_texts[summary.url] = derived.text

        if not item_summaries:
            excluded.append(ExcludedItem(url=url, reason="summary generation failed"))
            continue

    if not summaries:
        raise SystemExit("No valid articles after crawl/summarization.")

    log_stage("FILTER", f"included={len(summaries)} excluded={len(excluded)}")

    themes = group_themes(llm_config, summaries)
    log_stage("THEMES", f"count={len(themes)}")
    report = synthesize_report(
        config=llm_config,
        items=summaries,
        themes=themes,
        excluded=excluded,
    )
    log_stage("SYNTHESIZE", "draft_1 generated")

    allowed_urls = [item.url for item in summaries]
    draft_1_path = write_draft(str(output_path), run_id, "draft_1.md", report)

    eval_enabled = input_data.eval_enabled
    eval_1 = None
    eval_2 = None
    draft_2_path = None
    diff_path = None
    selected_report = report
    selected_name = "draft_1"
    selection_reason = "evaluation disabled" if not eval_enabled else "no revision"

    if eval_enabled:
        eval_1 = evaluate_report(llm_config, report, allowed_urls, source_texts)
        log_stage(
            "EVAL_DRAFT_1",
            f"score={eval_1.overall_score} rubric={eval_1.rubric.rubric_score} grounded={eval_1.groundedness.score}",
        )

        if needs_revision(eval_1):
            log_stage("REVISION", "creating draft_2")
            try:
                revised_report = revise_report(llm_config, report, eval_1, allowed_urls)
                draft_2_path = write_draft(str(output_path), run_id, "draft_2.md", revised_report)
                diff_path = write_diff(str(output_path), run_id, report, revised_report)
                log_stage("REVISION", "draft_2 generated")

                eval_2 = evaluate_report(llm_config, revised_report, allowed_urls, source_texts)
                log_stage(
                    "EVAL_DRAFT_2",
                    f"score={eval_2.overall_score} rubric={eval_2.rubric.rubric_score} grounded={eval_2.groundedness.score}",
                )

                if eval_2.overall_score > eval_1.overall_score:
                    selected_report = revised_report
                    selected_name = "draft_2"
                    selection_reason = "higher score"
                elif eval_2.overall_score < eval_1.overall_score:
                    selected_report = report
                    selected_name = "draft_1"
                    selection_reason = "higher score"
                else:
                    selected_report = revised_report
                    selected_name = "draft_2"
                    selection_reason = "tie -> revised"
            except Exception as exc:
                log_stage("REVISION", f"failed: {exc}")
                selected_report = report
                selected_name = "draft_1"
                selection_reason = "revision failed"
    else:
        log_stage("EVAL_SKIPPED", "disabled")

    log_stage("SELECT", f"{selected_name} reason={selection_reason}")

    report = selected_report

    issues = deterministic_checks(
        report_text=report,
        items=summaries,
        input_urls=input_data.urls,
        excluded=excluded,
    )

    found_urls = find_link_urls(report)
    unexpected = found_urls - set(allowed_urls)
    if unexpected:
        _write_artifacts(
            str(output_path),
            input_data=input_data,
            summaries=summaries,
            excluded=excluded,
            crawl_failures=crawl_failures,
            issues=[f"Unexpected sources in report: {', '.join(sorted(unexpected))}"],
            run_id=run_id,
            evaluation=_build_evaluation_payload(eval_1, eval_2, selected_name, selection_reason),
            drafts=_build_drafts_payload(draft_1_path, draft_2_path, diff_path),
            newsletter_splits=newsletter_splits,
        )
        raise SystemExit(f"Unexpected sources in report: {', '.join(sorted(unexpected))}")

    if issues:
        _write_artifacts(
            str(output_path),
            input_data=input_data,
            summaries=summaries,
            excluded=excluded,
            crawl_failures=crawl_failures,
            issues=issues,
            run_id=run_id,
            evaluation=_build_evaluation_payload(eval_1, eval_2, selected_name, selection_reason),
            drafts=_build_drafts_payload(draft_1_path, draft_2_path, diff_path),
            newsletter_splits=newsletter_splits,
        )
        raise SystemExit("Deterministic validation failed: " + "; ".join(issues))

    _write_report(str(output_path), report)
    _write_artifacts(
        str(output_path),
        input_data=input_data,
        summaries=summaries,
        excluded=excluded,
        crawl_failures=crawl_failures,
        issues=issues,
        run_id=run_id,
        evaluation=_build_evaluation_payload(eval_1, eval_2, selected_name, selection_reason),
        drafts=_build_drafts_payload(draft_1_path, draft_2_path, diff_path),
        newsletter_splits=newsletter_splits,
    )
    log_stage("WRITE_OUTPUT", f"report={output_path} artifacts=artifacts/run_{run_id}.json")


def _write_report(out_path: str, report: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _write_artifacts(
    out_path: str,
    input_data,
    summaries: List[SummaryItem],
    excluded: List[ExcludedItem],
    crawl_failures,
    issues: List[str],
    run_id: str,
    evaluation: dict,
    drafts: dict,
    newsletter_splits: List[dict],
) -> None:
    artifact_path = Path(out_path).parent.parent / "artifacts" / f"run_{run_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "input": {
            "urls": input_data.urls,
            "evaluation_enabled": input_data.eval_enabled,
        },
        "summaries": [
            {
                "url": item.url,
                "origin_url": item.origin_url,
                "source_name": item.source_name,
                "title": item.title,
                "date": format_date(item.date),
                "date_inferred": item.date_inferred,
                "summary_tr": item.summary_tr,
                "why_it_matters_tr": item.why_it_matters_tr,
                "tags": item.tags,
                "confidence": item.confidence,
            }
            for item in summaries
        ],
        "excluded": [{"url": ex.url, "reason": ex.reason} for ex in excluded],
        "crawl_failures": crawl_failures,
        "deterministic_issues": issues,
        "drafts": drafts,
        "evaluation": evaluation,
        "newsletter_splits": newsletter_splits,
    }

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_raw_text(out_path: str, crawl_items, run_id: str) -> None:
    raw_dir = Path(out_path).parent.parent / "artifacts" / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    for item in crawl_items:
        slug = slugify_url(item.url)
        file_path = raw_dir / f"{slug}.txt"
        file_path.write_text(item.text or "", encoding="utf-8")


def _build_drafts_payload(draft_1_path, draft_2_path, diff_path) -> dict:
    return {
        "draft_1_path": str(draft_1_path) if draft_1_path else None,
        "draft_2_path": str(draft_2_path) if draft_2_path else None,
        "diff_path": str(diff_path) if diff_path else None,
    }


def _build_evaluation_payload(eval_1, eval_2, selected_name: str, reason: str) -> dict:
    return {
        "draft_1": eval_result_to_dict(eval_1),
        "draft_2": eval_result_to_dict(eval_2) if eval_2 else None,
        "selected": selected_name,
        "reason": reason,
    }


def _write_split_items(out_path: str, run_id: str, origin_url: str, items: List[CrawlItem]) -> List[str]:
    base_dir = Path(out_path).parent.parent / "artifacts" / "splits" / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    origin_slug = slugify_url(origin_url)
    split_dir = base_dir / origin_slug
    split_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Also save a single concatenated file for easier viewing
    concat_path = split_dir / "_all_items_concat.txt"
    concat_content = [f"Source: {origin_url}\nTotal Items: {len(items)}\n{'='*50}\n"]

    paths: List[str] = []
    for idx, item in enumerate(items, start=1):
        filename = f"item_{idx:02d}.txt"
        path = split_dir / filename
        
        item_text = (
            f"Item {idx}\n"
            f"Title: {item.title or ''}\n"
            f"URL: {item.url}\n"
            f"{'-'*20}\n"
            f"{item.text or ''}\n"
        )
        
        # Write individual file
        path.write_text(item_text, encoding="utf-8")
        paths.append(str(path))
        
        # Append to consolidated file
        concat_content.append(item_text)
        concat_content.append(f"\n{'='*50}\n")
    
    # Save the consolidated file
    concat_path.write_text("\n".join(concat_content), encoding="utf-8")
    
    return paths


def _write_debug_input(out_path: str, run_id: str, item: CrawlItem) -> None:
    debug_dir = Path(out_path).parent.parent / "artifacts" / "debug_inputs" / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_url(item.url)
    path = debug_dir / f"{slug}.txt"
    content = f"Title: {item.title}\nURL: {item.url}\n\n{item.text}"
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
