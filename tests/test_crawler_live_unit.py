from pathlib import Path

from tests.test_crawler_live import resolve_live_crawl_input


def test_resolve_live_crawl_input_uses_default_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("CRAWL_TEST_INPUT", raising=False)

    assert resolve_live_crawl_input() == Path("inputs/links_05-02-2026.yaml")


def test_resolve_live_crawl_input_uses_env_override(monkeypatch):
    monkeypatch.setenv("CRAWL_TEST_INPUT", "inputs/custom_links.yaml")

    assert resolve_live_crawl_input() == Path("inputs/custom_links.yaml")
