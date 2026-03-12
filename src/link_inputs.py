from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ingest import load_input


@dataclass
class WriteLinksResult:
    path: Path
    captured_count: int
    added_count: int
    duplicate_count: int
    invalid_count: int
    evaluation: bool
    urls: list[str]


def write_links_input(
    path: str | Path,
    captured_urls: list[str],
    *,
    replace: bool = False,
    evaluation_default: bool = True,
) -> WriteLinksResult:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    captured_count = len(captured_urls)
    invalid_count = 0
    duplicate_count = 0

    existing_urls: list[str] = []
    evaluation = evaluation_default
    if output_path.exists() and not replace:
        existing_input = load_input(output_path)
        existing_urls = existing_input.urls
        evaluation = existing_input.eval_enabled

    seen = set(existing_urls if not replace else [])
    merged_urls = list(existing_urls if not replace else [])

    for raw_url in captured_urls:
        url = raw_url.strip()
        if not _is_supported_url(url):
            invalid_count += 1
            continue
        if url in seen:
            duplicate_count += 1
            continue
        seen.add(url)
        merged_urls.append(url)

    if replace:
        evaluation = evaluation_default

    output_path.write_text(_render_links_yaml(evaluation, merged_urls), encoding="utf-8")

    added_count = len(merged_urls) - len(existing_urls if not replace else [])
    return WriteLinksResult(
        path=output_path.resolve(),
        captured_count=captured_count,
        added_count=added_count,
        duplicate_count=duplicate_count,
        invalid_count=invalid_count,
        evaluation=evaluation,
        urls=merged_urls,
    )


def _render_links_yaml(evaluation: bool, urls: list[str]) -> str:
    lines = [f"evaluation: {'true' if evaluation else 'false'}", "urls:"]
    for url in urls:
        lines.append(f"  - {json.dumps(url, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _is_supported_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")
