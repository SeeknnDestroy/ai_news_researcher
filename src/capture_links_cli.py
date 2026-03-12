from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from .chrome_tabs import ChromeTabsError, capture_chrome_urls
from .ingest import InputError
from .link_inputs import write_links_input
from .storage_paths import dated_input_path

app = typer.Typer(help="Capture Google Chrome tabs into today's input YAML.")


def _default_output_path() -> Path:
    return dated_input_path(datetime.now().date())


@app.command()
def main(
    all_windows: bool = typer.Option(False, "--all-windows", help="Capture tabs from all Chrome windows."),
    replace: bool = typer.Option(False, "--replace", help="Replace the target YAML instead of appending."),
    output: Path = typer.Option(None, "--output", help="Target YAML path."),
    evaluation: bool = typer.Option(
        True,
        "--evaluation/--no-evaluation",
        help="Evaluation flag to use when creating a new file or replacing an existing one.",
    ),
) -> None:
    target_path = output or _default_output_path()

    try:
        captured_urls = capture_chrome_urls(all_windows=all_windows)
        result = write_links_input(
            target_path,
            captured_urls,
            replace=replace,
            evaluation_default=evaluation,
        )
    except (ChromeTabsError, InputError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    typer.echo(f"Target file: {result.path}")
    typer.echo(f"Captured URLs: {result.captured_count}")
    typer.echo(f"Added URLs: {result.added_count}")
    typer.echo(f"Skipped duplicates: {result.duplicate_count}")
    typer.echo(f"Skipped invalid: {result.invalid_count}")


if __name__ == "__main__":
    app()
