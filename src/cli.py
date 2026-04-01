from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import typer

from .llm import OpenAIConfig
from .application.pipeline import PipelineRequest, run_report_pipeline
from .config import get_settings
from .crawler import CrawlError
from .ingest import InputError
from .infrastructure.crawl_service import Crawl4AICrawlService
from .infrastructure.events import CompositeEventSink, ConsoleEventSink
from .infrastructure.llm_client import OpenAILLMClient
from .infrastructure.tracker_sink import TrackerEventSink
from .regression import run_regression_matrix


app = typer.Typer(help="GenAI weekly report generator (MVP1)")

@app.command()
def run(
    model: str = typer.Option("gpt-5.4-nano", help="OpenAI model name"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    max_concurrency: int = typer.Option(3, help="Max simultaneous crawls"),
    tracker: bool = typer.Option(True, "--tracker/--no-tracker", help="Start the local tracker UI."),
):
    """Generate a weekly report from a URL list."""
    asyncio.run(run_pipeline_async(model, temperature, max_concurrency, tracker))


@app.command("compare-regression")
def compare_regression(
    baseline_artifact: Path = typer.Option(
        Path("artifacts/run_12-03-2026_105525.json"),
        help="Historical baseline artifact path.",
    ),
    baseline_report: Path = typer.Option(
        Path("reports/2026-03/12-03-2026_weekly.md"),
        help="Historical baseline markdown report path.",
    ),
    live_input: Path = typer.Option(
        Path("inputs/2026-04/links_01-04-2026.yaml"),
        help="Live input YAML path for the current-code rerun.",
    ),
    models: str = typer.Option(
        "gpt-5.4-nano,gpt-5.4-mini",
        help="Comma-separated model list for current-code replay/live lanes.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Optional output directory for the regression bundle. Defaults to a temp workspace.",
    ),
    max_concurrency: int = typer.Option(3, help="Max simultaneous crawls for live lanes."),
):
    """Compare the March 12, 2026 baseline against current replay/live lanes."""
    model_list = [item.strip() for item in models.split(",") if item.strip()]
    result = asyncio.run(
        run_regression_matrix(
            baseline_artifact_path=baseline_artifact,
            baseline_report_path=baseline_report,
            live_input_path=live_input,
            models=model_list,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        )
    )
    typer.echo(f"Regression bundle: {result.output_dir}")
    typer.echo(f"JSON summary: {result.json_path}")
    typer.echo(f"Markdown summary: {result.markdown_path}")


async def run_pipeline_async(model: str, temperature: float, max_concurrency: int, tracker: bool = True):
    settings = get_settings()
    llm_config = OpenAIConfig(
        model=model,
        temperature=temperature,
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

    sinks = [ConsoleEventSink()]
    if tracker:
        from .tracker.server import start_server_in_background

        tracker_state = start_server_in_background()
        sinks.append(TrackerEventSink(tracker_state))

    try:
        result = await run_report_pipeline(
            llm_client=OpenAILLMClient(llm_config),
            crawl_service=Crawl4AICrawlService(),
            request=PipelineRequest(
                target_date=datetime.now().date(),
                max_concurrency=max_concurrency,
            ),
            event_sink=CompositeEventSink(*sinks),
        )
        _print_usage_summary(result.metadata.llm_usage)
        return result
    except (InputError, CrawlError) as exc:
        raise SystemExit(str(exc))


def _print_usage_summary(usage_summary: dict[str, object]) -> None:
    totals = usage_summary.get("totals")
    if not isinstance(totals, dict):
        return

    print(
        "LLM usage | "
        f"requests={totals.get('request_count', 0)} "
        f"input={totals.get('input_tokens', 0)} "
        f"output={totals.get('output_tokens', 0)} "
        f"cached={totals.get('cached_input_tokens', 0)} "
        f"reasoning={totals.get('reasoning_tokens', 0)} "
        f"total={totals.get('total_tokens', 0)}",
        flush=True,
    )

    _print_usage_group("LLM usage by role", usage_summary.get("by_role"))
    _print_usage_group("LLM usage by task", usage_summary.get("by_task"))


def _print_usage_group(title: str, usage_group: object) -> None:
    if not isinstance(usage_group, dict) or not usage_group:
        return

    print(title, flush=True)
    for key in sorted(usage_group):
        bucket = usage_group.get(key)
        if not isinstance(bucket, dict):
            continue
        print(
            f"  {key}: "
            f"requests={bucket.get('request_count', 0)} "
            f"input={bucket.get('input_tokens', 0)} "
            f"output={bucket.get('output_tokens', 0)} "
            f"cached={bucket.get('cached_input_tokens', 0)} "
            f"reasoning={bucket.get('reasoning_tokens', 0)} "
            f"total={bucket.get('total_tokens', 0)}",
            flush=True,
        )


if __name__ == "__main__":
    app()
