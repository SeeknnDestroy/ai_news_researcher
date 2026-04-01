from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli import app
from src.regression import (
    ReplayCrawlService,
    RunSnapshot,
    compare_snapshots,
    run_regression_matrix,
)


def _write_debug_input(path: Path, *, title: str, url: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Title: {title}\n"
        f"URL: {url}\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_replay_crawl_service_reconstructs_items_and_failures(tmp_path: Path):
    baseline_artifact = tmp_path / "baseline.json"
    debug_dir = tmp_path / "debug_inputs"
    kept_url = "https://example.com/kept"
    failed_url = "https://example.com/failed"

    baseline_artifact.write_text(
        json.dumps(
            {
                "input": {
                    "urls": [kept_url, failed_url],
                    "evaluation_enabled": True,
                },
                "crawl_failures": [{"url": failed_url, "reason": "timeout"}],
                "summaries": [],
                "excluded": [],
                "newsletter_splits": [],
                "drafts": {
                    "draft_outline": {
                        "report_title": "Baseline",
                        "introduction_commentary": "Intro",
                        "themes": [],
                    }
                },
                "evaluation": {
                    "critique": "",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_debug_input(
        debug_dir / "kept.txt",
        title="Kept article",
        url=kept_url,
        body="March 5, 2026\nA reconstructed article body.",
    )

    service = ReplayCrawlService.from_paths(baseline_artifact, debug_dir)
    result = asyncio.run(service.crawl([kept_url, failed_url], max_concurrency=2))

    assert [item.url for item in result.items] == [kept_url]
    assert result.items[0].title == "Kept article"
    assert result.failures[0].url == failed_url
    assert result.failures[0].reason == "timeout"


def test_compare_snapshots_reports_regression_and_heading_deltas():
    baseline = RunSnapshot(
        lane_id="baseline_xai",
        mode="historical",
        model="xai",
        family="xai",
        input_urls=["https://example.com/a", "https://example.com/b"],
        evaluation_enabled=True,
        crawl_ok_count=2,
        crawl_failures=[],
        summaries=[
            {
                "url": "https://example.com/a",
                "title": "Alpha",
                "date": "05 March 2026",
                "date_inferred": False,
                "confidence": 0.9,
            },
            {
                "url": "https://example.com/b",
                "title": "Beta",
                "date": "06 March 2026",
                "date_inferred": False,
                "confidence": 0.8,
            },
        ],
        excluded=[],
        newsletter_splits=[],
        outline={
            "report_title": "Baseline title",
            "themes": [
                {
                    "theme_name": "1. Agents",
                    "articles": [
                        {
                            "heading": "Alpha heading",
                            "primary_url": "https://example.com/a",
                            "news_urls_included": ["https://example.com/a"],
                        },
                        {
                            "heading": "Beta heading",
                            "primary_url": "https://example.com/b",
                            "news_urls_included": ["https://example.com/b"],
                        },
                    ],
                }
            ],
        },
        evaluation={"passes_criteria": True, "critique": "ok", "specific_fixes_required": []},
        revision_count=0,
        report_text="# Baseline\n\n## 1. Agents\n\n### Alpha heading\n\n### Beta heading\n",
        report_sections=["1. Agents"],
        article_headings=["Alpha heading", "Beta heading"],
        artifact_path=None,
        report_path=None,
        error=None,
    )
    candidate = RunSnapshot(
        lane_id="replay_gpt_5_4_nano",
        mode="replay",
        model="gpt-5.4-nano",
        family="openai",
        input_urls=["https://example.com/a", "https://example.com/b"],
        evaluation_enabled=True,
        crawl_ok_count=1,
        crawl_failures=[{"url": "https://example.com/b", "reason": "timeout"}],
        summaries=[
            {
                "url": "https://example.com/a",
                "title": "Alpha rewritten",
                "date": "05 March 2026",
                "date_inferred": True,
                "confidence": 0.55,
            }
        ],
        excluded=[{"url": "https://example.com/b", "reason": "timeout", "stage": "crawl"}],
        newsletter_splits=[],
        outline={
            "report_title": "Candidate title",
            "themes": [
                {
                    "theme_name": "1. Agents",
                    "articles": [
                        {
                            "heading": "Alpha heading",
                            "primary_url": "https://example.com/a",
                            "news_urls_included": ["https://example.com/a"],
                        },
                    ],
                }
            ],
        },
        evaluation={"passes_criteria": False, "critique": "needs work", "specific_fixes_required": ["Fix headings"]},
        revision_count=1,
        report_text="# Candidate\n\n## 1. Agents\n\n### Alpha heading\n",
        report_sections=["1. Agents"],
        article_headings=["Alpha heading"],
        artifact_path=None,
        report_path=None,
        error=None,
    )

    comparison = compare_snapshots(baseline, candidate)

    assert comparison["summary"]["candidate_count"] == 1
    assert comparison["crawl"]["failure_delta"] == 1
    assert comparison["report"]["removed_article_headings"] == ["Beta heading"]
    assert comparison["workflow"]["evaluation_pass_changed"] is True
    assert comparison["verdict"]["regressed"] is True


def test_run_regression_matrix_writes_multimodel_outputs(tmp_path: Path, monkeypatch):
    baseline_artifact = tmp_path / "run_12-03-2026_105525.json"
    baseline_report = tmp_path / "12-03-2026_weekly.md"
    live_input = tmp_path / "inputs" / "2026-04" / "links_01-04-2026.yaml"

    baseline_artifact.write_text(
        json.dumps(
            {
                "input": {
                    "urls": ["https://example.com/a"],
                    "evaluation_enabled": True,
                },
                "crawl_failures": [],
                "summaries": [
                    {
                        "url": "https://example.com/a",
                        "origin_url": "https://example.com/a",
                        "source_name": "Example",
                        "title": "Baseline title",
                        "date": "05 March 2026",
                        "date_inferred": False,
                        "summary_tr": "Summary",
                        "why_it_matters_tr": "Why",
                        "tags": ["agentic"],
                        "confidence": 0.9,
                    }
                ],
                "excluded": [],
                "newsletter_splits": [],
                "drafts": {
                    "draft_outline": {
                        "report_title": "Baseline title",
                        "introduction_commentary": "Intro",
                        "themes": [
                            {
                                "theme_name": "1. Agents",
                                "theme_commentary": "Commentary",
                                "articles": [
                                    {
                                        "heading": "Alpha heading",
                                        "primary_url": "https://example.com/a",
                                        "news_urls_included": ["https://example.com/a"],
                                        "content_plan": "Plan",
                                    }
                                ],
                            }
                        ],
                    }
                },
                "evaluation": {
                    "critique": "ok",
                    "specific_fixes_required": [],
                    "passes_criteria": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    baseline_report.write_text("# Baseline\n\n## 1. Agents\n\n### Alpha heading\n", encoding="utf-8")
    live_input.parent.mkdir(parents=True, exist_ok=True)
    live_input.write_text('evaluation: true\nurls:\n  - "https://example.com/a"\n', encoding="utf-8")

    async def fake_execute_lane(*, lane_id, mode, model, baseline, lane_output_dir, **kwargs):
        lane_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = lane_output_dir / "artifact.json"
        report_path = lane_output_dir / "report.md"
        artifact_path.write_text(
            json.dumps(
                {
                    "input": {
                        "path": str(lane_output_dir / "inputs.yaml"),
                        "urls": baseline.input_urls,
                        "evaluation_enabled": True,
                    },
                    "crawl": {"ok_count": 1, "failure_count": 0, "failures": []},
                    "summaries": [
                        {
                            "url": "https://example.com/a",
                            "origin_url": "https://example.com/a",
                            "source_name": "Example",
                            "title": f"{model} title",
                            "date": "05 March 2026",
                            "date_inferred": False,
                            "summary_tr": "Summary",
                            "why_it_matters_tr": "Why",
                            "tags": ["agentic"],
                            "confidence": 0.8 if model.endswith("nano") else 0.7,
                        }
                    ],
                    "excluded": [],
                    "newsletter_splits": [],
                    "workflow": {
                        "outline": {
                            "report_title": f"{model} report",
                            "introduction_commentary": "Intro",
                            "themes": [
                                {
                                    "theme_name": "1. Agents",
                                    "theme_commentary": "Commentary",
                                    "articles": [
                                            {
                                                "heading": "Alpha heading",
                                                "primary_url": "https://example.com/a",
                                                "news_urls_included": ["https://example.com/a"],
                                                "content_plan": "Plan",
                                            }
                                    ],
                                }
                            ],
                        },
                        "evaluation": {
                            "critique": "ok",
                            "specific_fixes_required": [],
                            "passes_criteria": True,
                        },
                        "critique": "ok",
                        "revision_count": 0,
                        "critique_history": [],
                    },
                    "metadata": {
                        "run_id": lane_id,
                        "started_at": "2026-04-01T10:00:00",
                        "stage_timings": {"crawl": 0.1},
                        "validation_failures": [],
                        "retries": {"draft_revision": 0},
                        "fallbacks": [],
                        "llm_usage": {},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report_path.write_text(f"# {model}\n\n## 1. Agents\n\n### Alpha heading\n", encoding="utf-8")
        return lane_output_dir, artifact_path, report_path, None

    monkeypatch.setattr("src.regression._execute_lane", fake_execute_lane)

    result = asyncio.run(
        run_regression_matrix(
            baseline_artifact_path=baseline_artifact,
            baseline_report_path=baseline_report,
            live_input_path=live_input,
            models=["gpt-5.4-nano", "gpt-5.4-mini"],
            output_dir=tmp_path / "out",
            max_concurrency=2,
        )
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert (tmp_path / "out" / "lanes" / "replay_gpt_5_4_nano").exists()
    assert (tmp_path / "out" / "lanes" / "replay_gpt_5_4_mini").exists()
    assert (tmp_path / "out" / "lanes" / "live_gpt_5_4_nano").exists()
    assert (tmp_path / "out" / "lanes" / "live_gpt_5_4_mini").exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "nano_vs_mini_replay_delta" in payload["cross_model_deltas"]
    assert "historical_xai_vs_nano_delta" in payload["cross_model_deltas"]


def test_compare_regression_cli_invokes_harness(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    output_dir = tmp_path / "bundle"

    async def fake_run_regression_matrix(**kwargs):
        del kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "regression_summary.json"
        markdown_path = output_dir / "regression_summary.md"
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# Summary\n", encoding="utf-8")
        result = type("Result", (), {})()
        result.output_dir = output_dir
        result.json_path = json_path
        result.markdown_path = markdown_path
        return result

    monkeypatch.setattr("src.cli.run_regression_matrix", fake_run_regression_matrix)

    result = runner.invoke(
        app,
        [
            "compare-regression",
            "--baseline-artifact",
            str(tmp_path / "baseline.json"),
            "--baseline-report",
            str(tmp_path / "baseline.md"),
            "--live-input",
            str(tmp_path / "live.yaml"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert f"Regression bundle: {output_dir}" in result.stdout
    assert f"JSON summary: {output_dir / 'regression_summary.json'}" in result.stdout
    assert f"Markdown summary: {output_dir / 'regression_summary.md'}" in result.stdout
