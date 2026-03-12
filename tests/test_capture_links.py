from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.capture_links_cli import app
from src.chrome_tabs import ChromeTabsError, _parse_osascript_output, capture_chrome_urls
from src.link_inputs import write_links_input
from src.storage_paths import dated_input_path, dated_report_path, resolve_input_path


def test_parse_osascript_output_filters_non_web_tabs():
    raw = "\n".join(
        [
            "https://example.com/article",
            "chrome://newtab/",
            "   ",
            "https://example.com/another",
            "chrome-extension://abc/popup.html",
            "http://example.org/post",
        ]
    )

    assert _parse_osascript_output(raw) == [
        "https://example.com/article",
        "https://example.com/another",
        "http://example.org/post",
    ]


def test_capture_chrome_urls_raises_clear_error_on_permission_denied(monkeypatch):
    def fake_run_osascript(lines: list[str]) -> str:
        raise ChromeTabsError(
            "Automation permission denied. Allow your terminal app to control Google Chrome in System Settings."
        )

    monkeypatch.setattr("src.chrome_tabs._run_osascript", fake_run_osascript)

    with pytest.raises(ChromeTabsError, match="Automation permission denied"):
        capture_chrome_urls()


def test_write_links_input_appends_and_dedupes_urls(tmp_path: Path):
    path = tmp_path / "inputs" / "links_13-03-2026.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'evaluation: false\nurls:\n  - "https://existing.example/a"\n  - "https://existing.example/b"\n',
        encoding="utf-8",
    )

    result = write_links_input(
        path,
        [
            "https://existing.example/b",
            "https://new.example/c",
            "https://new.example/d",
        ],
        replace=False,
        evaluation_default=True,
    )

    assert result.added_count == 2
    assert result.duplicate_count == 1
    assert result.invalid_count == 0
    assert result.evaluation is False
    assert path.read_text(encoding="utf-8") == (
        'evaluation: false\n'
        "urls:\n"
        '  - "https://existing.example/a"\n'
        '  - "https://existing.example/b"\n'
        '  - "https://new.example/c"\n'
        '  - "https://new.example/d"\n'
    )


def test_write_links_input_creates_new_file_with_default_evaluation(tmp_path: Path):
    path = tmp_path / "inputs" / "links_13-03-2026.yaml"

    result = write_links_input(
        path,
        ["https://fresh.example/a"],
        replace=False,
        evaluation_default=True,
    )

    assert result.added_count == 1
    assert result.duplicate_count == 0
    assert result.evaluation is True
    assert path.read_text(encoding="utf-8") == (
        'evaluation: true\n'
        "urls:\n"
        '  - "https://fresh.example/a"\n'
    )


def test_write_links_input_replace_mode_rebuilds_file(tmp_path: Path):
    path = tmp_path / "inputs" / "links_13-03-2026.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'evaluation: false\nurls:\n  - "https://existing.example/a"\n',
        encoding="utf-8",
    )

    result = write_links_input(
        path,
        ["https://replacement.example/a", "https://replacement.example/a"],
        replace=True,
        evaluation_default=True,
    )

    assert result.added_count == 1
    assert result.duplicate_count == 1
    assert result.evaluation is True
    assert path.read_text(encoding="utf-8") == (
        'evaluation: true\n'
        "urls:\n"
        '  - "https://replacement.example/a"\n'
    )


def test_dated_paths_use_monthly_folders():
    target_date = date(2026, 3, 13)

    assert dated_input_path(target_date) == Path("inputs") / "2026-03" / "links_13-03-2026.yaml"
    assert dated_report_path(target_date) == Path("reports") / "2026-03" / "13-03-2026_weekly.md"


def test_resolve_input_path_prefers_monthly_path(tmp_path: Path):
    monthly_path = tmp_path / "inputs" / "2026-03" / "links_13-03-2026.yaml"
    legacy_path = tmp_path / "inputs" / "links_13-03-2026.yaml"
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_path.write_text('evaluation: true\nurls:\n  - "https://monthly.example/a"\n', encoding="utf-8")
    legacy_path.write_text('evaluation: true\nurls:\n  - "https://legacy.example/a"\n', encoding="utf-8")

    resolved = resolve_input_path(date(2026, 3, 13), base_dir=tmp_path)

    assert resolved == monthly_path


def test_resolve_input_path_falls_back_to_legacy_flat_file(tmp_path: Path):
    legacy_path = tmp_path / "inputs" / "links_13-03-2026.yaml"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text('evaluation: true\nurls:\n  - "https://legacy.example/a"\n', encoding="utf-8")

    resolved = resolve_input_path(date(2026, 3, 13), base_dir=tmp_path)

    assert resolved == legacy_path


def test_capture_links_cli_writes_summary_and_file(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    output_path = tmp_path / "inputs" / "links_13-03-2026.yaml"

    monkeypatch.setattr(
        "src.capture_links_cli.capture_chrome_urls",
        lambda all_windows=False: [
            "https://captured.example/a",
            "https://captured.example/a",
            "chrome://settings/",
            "https://captured.example/b",
        ],
    )

    result = runner.invoke(app, ["--output", str(output_path)])

    assert result.exit_code == 0
    assert f"Target file: {output_path}" in result.stdout
    assert "Captured URLs: 4" in result.stdout
    assert "Added URLs: 2" in result.stdout
    assert "Skipped duplicates: 1" in result.stdout
    assert "Skipped invalid: 1" in result.stdout
    assert output_path.read_text(encoding="utf-8") == (
        'evaluation: true\n'
        "urls:\n"
        '  - "https://captured.example/a"\n'
        '  - "https://captured.example/b"\n'
    )


def test_capture_links_cli_default_output_uses_month_folder(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    class FixedDateTime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 9, 30, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.capture_links_cli.datetime", FixedDateTime)
    monkeypatch.setattr(
        "src.capture_links_cli.capture_chrome_urls",
        lambda all_windows=False: ["https://captured.example/a"],
    )

    result = runner.invoke(app, [])

    expected_path = tmp_path / "inputs" / "2026-03" / "links_13-03-2026.yaml"
    assert result.exit_code == 0
    assert f"Target file: {expected_path}" in result.stdout
    assert expected_path.exists()


def test_capture_links_cli_returns_nonzero_on_capture_error(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "src.capture_links_cli.capture_chrome_urls",
        lambda all_windows=False: (_ for _ in ()).throw(ChromeTabsError("Google Chrome is not running.")),
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "Google Chrome is not running." in result.stdout
