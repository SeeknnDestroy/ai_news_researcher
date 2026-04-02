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
        f"Title: {title}\nURL: {url}\n\n{body}\n",
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


def test_compare_snapshots_reports_story_card_and_outline_regressions():
    baseline = RunSnapshot(
        lane_id="baseline_xai",
        mode="historical",
        model="xai",
        family="xai",
        input_urls=["https://example.com/a", "https://example.com/b"],
        evaluation_enabled=True,
        crawl_ok_count=2,
        crawl_failures=[],
        story_cards=[
            {
                "url": "https://example.com/a",
                "story_title_tr": "Alpha",
                "published_at": "05 March 2026",
                "published_at_inferred": False,
                "confidence": 0.9,
            },
            {
                "url": "https://example.com/b",
                "story_title_tr": "Beta",
                "published_at": "06 March 2026",
                "published_at_inferred": False,
                "confidence": 0.8,
            },
        ],
        story_units=[
            {"story_unit_id": "story-a", "primary_url": "https://example.com/a"},
            {"story_unit_id": "story-b", "primary_url": "https://example.com/b"},
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
        lane_id="replay_routed_openai",
        mode="replay",
        model="routed_openai",
        family="openai",
        input_urls=["https://example.com/a", "https://example.com/b"],
        evaluation_enabled=True,
        crawl_ok_count=1,
        crawl_failures=[{"url": "https://example.com/b", "reason": "timeout"}],
        story_cards=[
            {
                "url": "https://example.com/a",
                "story_title_tr": "Alpha rewritten",
                "published_at": "05 March 2026",
                "published_at_inferred": True,
                "confidence": 0.55,
            }
        ],
        story_units=[{"story_unit_id": "story-a", "primary_url": "https://example.com/a"}],
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
        evaluation={
            "passes_criteria": False,
            "critique": "needs work",
            "specific_fixes_required": ["Fix headings"],
        },
        revision_count=1,
        report_text="# Candidate\n\n## 1. Agents\n\n### Alpha heading\n",
        report_sections=["1. Agents"],
        article_headings=["Alpha heading"],
        artifact_path=None,
        report_path=None,
        error=None,
    )

    comparison = compare_snapshots(baseline, candidate)

    assert comparison["story_cards"]["candidate_count"] == 1
    assert comparison["crawl"]["failure_delta"] == 1
    assert comparison["story_cards"]["missing_story_card_urls"] == ["https://example.com/b"]
    assert comparison["workflow"]["missing_outline_url_count"] == 0
    assert comparison["report"]["removed_article_headings"] == ["Beta heading"]
    assert comparison["verdict"]["regressed"] is True


def test_run_regression_matrix_writes_routed_outputs(tmp_path: Path, monkeypatch):
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
    baseline_report.write_text(
        "# Baseline\n\n## 1. Agents\n\n### Alpha heading\n", encoding="utf-8"
    )
    live_input.parent.mkdir(parents=True, exist_ok=True)
    live_input.write_text(
        'evaluation: true\nurls:\n  - "https://example.com/a"\n', encoding="utf-8"
    )

    async def fake_execute_lane(*, lane_id, mode, lane_output_dir, **kwargs):
        lane_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = lane_output_dir / "artifact.json"
        report_path = lane_output_dir / "report.md"
        artifact_path.write_text(
            json.dumps(
                {
                    "input": {
                        "path": str(lane_output_dir / "inputs.yaml"),
                        "urls": ["https://example.com/a"],
                        "evaluation_enabled": True,
                    },
                    "crawl": {
                        "ok_count": 1,
                        "failure_count": 0,
                        "failures": [],
                    },
                    "story_cards": [
                        {
                            "url": "https://example.com/a",
                            "story_title_tr": f"{mode} title",
                            "published_at": "05 March 2026",
                            "published_at_inferred": False,
                            "confidence": 0.8,
                        }
                    ],
                    "story_units": [
                        {
                            "story_unit_id": "story-a",
                            "primary_url": "https://example.com/a",
                            "news_urls_included": ["https://example.com/a"],
                        }
                    ],
                    "excluded": [],
                    "newsletter_splits": [],
                    "workflow": {
                        "theme_plan": {
                            "report_title": "Routed report",
                            "themes": [{"theme_name": "1. Agents", "story_unit_ids": ["story-a"]}],
                        },
                        "outline": {
                            "report_title": "Routed report",
                            "themes": [
                                {
                                    "theme_name": "1. Agents",
                                    "articles": [
                                        {
                                            "heading": f"{mode} heading",
                                            "primary_url": "https://example.com/a",
                                            "news_urls_included": ["https://example.com/a"],
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
                        "revision_count": 0,
                        "critique_history": [],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            f"# {mode}\n\n## 1. Agents\n\n### {mode} heading\n", encoding="utf-8"
        )
        return lane_output_dir, artifact_path, report_path, None

    monkeypatch.setattr("src.regression._execute_lane", fake_execute_lane)

    result = asyncio.run(
        run_regression_matrix(
            baseline_artifact_path=baseline_artifact,
            baseline_report_path=baseline_report,
            live_input_path=live_input,
            output_dir=tmp_path / "bundle",
        )
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert result.output_dir == tmp_path / "bundle"
    assert len(payload["lanes"]) == 2
    assert payload["lanes"][0]["model"] == "routed_openai"
    assert payload["baseline"]["lane_id"] == "historical_xai"


def test_compare_regression_cli_writes_bundle(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "bundle"

    class FakeResult:
        def __init__(self) -> None:
            self.output_dir = output_dir
            self.json_path = output_dir / "regression_summary.json"
            self.markdown_path = output_dir / "regression_summary.md"

    async def fake_run_regression_matrix(**kwargs):
        del kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "regression_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "regression_summary.md").write_text("# Summary", encoding="utf-8")
        return FakeResult()

    monkeypatch.setattr("src.cli.run_regression_matrix", fake_run_regression_matrix)

    runner = CliRunner()
    result = runner.invoke(app, ["compare-regression", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert f"Regression bundle: {output_dir}" in result.stdout
    assert f"JSON summary: {output_dir / 'regression_summary.json'}" in result.stdout
    assert f"Markdown summary: {output_dir / 'regression_summary.md'}" in result.stdout
