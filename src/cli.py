from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List
import typer

from .config import CrawlItem, ExcludedItem, SummaryItem
from .crawler import CrawlError, crawl_urls_async
from .date_extract import extract_date
from .ingest import InputError, load_input
from .llm import XAIConfig
from .summarize import summarize_article_async
from .themes import group_themes
from .newsletter import split_newsletter_items_async
from .agents.draft_agent import generate_draft_outline
from .agents.judge_agent import evaluate_draft_outline
from .agents.final_report_agent import generate_final_report
from .utils import format_date, slugify_url, log_stage
from .drafts import write_draft, write_diff


app = typer.Typer(help="GenAI weekly report generator (MVP1)")

@app.command()
def run(
    model: str = typer.Option("grok-4-1-fast-reasoning", help="xAI model name"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    max_concurrency: int = typer.Option(3, help="Max simultaneous crawls")
):
    """Generate a weekly report from a URL list."""
    from .tracker.server import start_server_in_background
    start_server_in_background()
    asyncio.run(run_pipeline_async(model, temperature, max_concurrency))


async def run_pipeline_async(model: str, temperature: float, max_concurrency: int) -> None:
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
        model=model,
        temperature=temperature,
    )

    log_stage("CRAWL", f"urls={len(input_data.urls)}")
    try:
        crawl_items, crawl_failures = await crawl_urls_async(
            input_data.urls, max_concurrency=max_concurrency
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

    async def process_url(idx: int, url: str):
        if url in failure_map:
            reason = failure_map.get(url, "crawl failed")
            excluded.append(ExcludedItem(url=url, reason=reason))
            return

        item = crawl_map.get(url)
        if not item:
            excluded.append(ExcludedItem(url=url, reason="crawl result missing"))
            return

        date_result = extract_date(item.metadata, item.text, item.url)

        derived_items = await split_newsletter_items_async(llm_config, item, max_items=6)
        if len(derived_items) > 1:
            log_stage("SPLIT_NEWSLETTER", f"{url} -> {len(derived_items)} items")
            split_paths = _write_split_items(str(output_path), run_id, url, derived_items)
            newsletter_splits.append({"origin_url": url, "paths": split_paths})
            if split_paths:
                log_stage("SPLIT_SAVE", f"origin={url} dir={Path(split_paths[0]).parent} files={len(split_paths)}")

        async def process_derived(derived, sub_index):
            _write_debug_input(str(output_path), run_id, derived)
            summary = await summarize_article_async(llm_config, derived)
            if summary:
                summary.date = date_result.value
                summary.date_inferred = date_result.inferred
            return summary, derived

        tasks = [process_derived(d, i) for i, d in enumerate(derived_items, start=1)]
        results = await asyncio.gather(*tasks)

        item_summaries = []
        for summary, derived in results:
            if summary and summary.summary_tr and summary.why_it_matters_tr:
                summaries.append(summary)
                item_summaries.append(summary)
                source_texts[summary.url] = derived.text

        if not item_summaries:
            excluded.append(ExcludedItem(url=url, reason="summary generation failed"))

    await asyncio.gather(*(process_url(i, u) for i, u in enumerate(input_data.urls, start=1)))

    if not summaries:
        raise SystemExit("No valid articles after crawl/summarization.")

    log_stage("FILTER", f"included={len(summaries)} excluded={len(excluded)}")

    # 1. First Draft / Outline Generation
    draft_outline = await generate_draft_outline(llm_config, summaries)
    
    # 2. Judge Evaluation
    max_retries = input_data.eval_enabled and 1 or 0
    retries = 0
    critique = ""
    previous_critiques_history = ""
    
    while retries <= max_retries:
        eval_result = await evaluate_draft_outline(llm_config, draft_outline, previous_critiques_history)
        passes = eval_result.get("passes_criteria", False)
        critique = eval_result.get("critique", "")
        issues = eval_result.get("specific_fixes_required", [])
        
        log_stage("JUDGING", f"Attempt {retries + 1}/{max_retries + 1}: Pass={passes}, Critique={critique}, Issues={issues}")
        
        if passes or retries >= max_retries:
            break
            
        log_stage("REVISION", f"Draft rejected. Retrying Draft generation with feedback...")
        
        combined_critique = critique
        if issues:
            combined_critique += "\n\nSpecific fixes required:\n" + "\n".join(f"- {i}" for i in issues)
            
        previous_critiques_history += f"Attempt {retries + 1} Critique:\n{combined_critique}\n\n"

        draft_outline = await generate_draft_outline(llm_config, summaries, critique=combined_critique, previous_draft=draft_outline)
        retries += 1
        
    # 3. Final Report Generation
    report = await generate_final_report(
        config=llm_config,
        outline=draft_outline,
        summaries=summaries,
        excluded=excluded,
        critique=critique
    )
    log_stage("FINAL_AGENT", "Final report generated")

    _write_report(str(output_path), report)
    
    # Write artifacts summary
    final_issues = eval_result.get("specific_fixes_required", []) if not eval_result.get("passes_criteria", True) else []
    if not eval_result.get("passes_criteria", True) and not final_issues:
        final_issues = ["Draft failed judging criteria"]
        
    _write_artifacts(
        str(output_path),
        input_data=input_data,
        summaries=summaries,
        excluded=excluded,
        crawl_failures=crawl_failures,
        issues=final_issues,
        run_id=run_id,
        evaluation=eval_result,
        drafts={"draft_outline": draft_outline},
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
    app()
