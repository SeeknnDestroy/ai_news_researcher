from __future__ import annotations

import asyncio
from datetime import datetime

import typer

from .llm import XAIConfig
from .application.pipeline import PipelineRequest, run_report_pipeline
from .config import get_settings
from .crawler import CrawlError
from .ingest import InputError
from .infrastructure.crawl_service import Crawl4AICrawlService
from .infrastructure.events import CompositeEventSink, ConsoleEventSink
from .infrastructure.llm_client import XAILLMClient
from .infrastructure.tracker_sink import TrackerEventSink


app = typer.Typer(help="GenAI weekly report generator (MVP1)")

@app.command()
def run(
    model: str = typer.Option("grok-4-1-fast-reasoning", help="xAI model name"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    max_concurrency: int = typer.Option(3, help="Max simultaneous crawls"),
    tracker: bool = typer.Option(True, "--tracker/--no-tracker", help="Start the local tracker UI."),
):
    """Generate a weekly report from a URL list."""
    asyncio.run(run_pipeline_async(model, temperature, max_concurrency, tracker))


async def run_pipeline_async(model: str, temperature: float, max_concurrency: int, tracker: bool = True):
    settings = get_settings()
    llm_config = XAIConfig(
        model=model,
        temperature=temperature,
        api_key=settings.xai_api_key,
        base_url=settings.xai_base_url,
        timeout_s=settings.xai_timeout_s,
    )

    sinks = [ConsoleEventSink()]
    if tracker:
        from .tracker.server import start_server_in_background

        tracker_state = start_server_in_background()
        sinks.append(TrackerEventSink(tracker_state))

    try:
        return await run_report_pipeline(
            llm_client=XAILLMClient(llm_config),
            crawl_service=Crawl4AICrawlService(),
            request=PipelineRequest(
                target_date=datetime.now().date(),
                max_concurrency=max_concurrency,
            ),
            event_sink=CompositeEventSink(*sinks),
        )
    except (InputError, CrawlError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    app()
