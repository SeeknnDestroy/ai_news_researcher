import os
from pathlib import Path

import pytest

from src.crawler import crawl_urls
from src.ingest import load_input


def resolve_live_crawl_input() -> Path:
    return Path(os.getenv("CRAWL_TEST_INPUT", "inputs/links_05-02-2026.yaml"))


@pytest.mark.integration
def test_crawl_live_urls_and_print_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))
    input_path = resolve_live_crawl_input()
    if not input_path.exists():
        pytest.skip(f"Live crawl input file not found: {input_path}")

    limit = int(os.getenv("CRAWL_TEST_LIMIT", "5"))
    output_dir = Path(os.getenv("CRAWL_TEST_OUTPUT", "artifacts/raw_live"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_data = load_input(input_path)
    urls = input_data.urls[:limit]

    items, failures = crawl_urls(urls, max_concurrency=2)

    assert not failures
    assert len(items) == len(urls)

    for item in items:
        safe_name = (
            item.url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")
            .replace("?", "_")
            .replace("&", "_")
            .replace("=", "_")
        )
        output_path = output_dir / f"{safe_name}.md"
        output_path.write_text(item.text, encoding="utf-8")
        preview = item.text[:800].replace("\n", " ").strip()
        print(f"\nURL: {item.url}\nTITLE: {item.title}\nTEXT: {preview}\n")
