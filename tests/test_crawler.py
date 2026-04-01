from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from src.crawler import _extract_pdf_text, crawl_urls


class _FakeMarkdown:
    def __init__(self, *, fit_markdown: str = "", raw_markdown: str = "", markdown: str = "") -> None:
        self.fit_markdown = fit_markdown
        self.raw_markdown = raw_markdown
        self.markdown = markdown


class _FakeCrawlResult:
    def __init__(
        self,
        *,
        url: str,
        success: bool,
        title: str | None = None,
        fit_markdown: str = "",
        raw_markdown: str = "",
        cleaned_html: str = "",
        metadata: dict[str, object] | None = None,
        error_message: str = "",
        status_code: int | None = None,
        redirected_url: str | None = None,
    ) -> None:
        self.url = url
        self.success = success
        self.title = title
        self.markdown = _FakeMarkdown(
            fit_markdown=fit_markdown,
            raw_markdown=raw_markdown,
        )
        self.cleaned_html = cleaned_html
        self.metadata = metadata or {}
        self.error_message = error_message
        self.status_code = status_code
        self.redirected_url = redirected_url


class _FakeBrowserConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCrawlerRunConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeDefaultMarkdownGenerator:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakePruningContentFilter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeMemoryAdaptiveDispatcher:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.max_session_permit = kwargs.get("max_session_permit")


class _FakeUndetectedAdapter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeAsyncPlaywrightCrawlerStrategy:
    def __init__(self, *, browser_config=None, browser_adapter=None) -> None:
        self.browser_config = browser_config
        self.browser_adapter = browser_adapter


class _FakeAsyncWebCrawler:
    instances: list["_FakeAsyncWebCrawler"] = []
    arun_many_calls: list[dict[str, object]] = []

    def __init__(self, *, config=None, crawler_strategy=None) -> None:
        self.config = config
        self.crawler_strategy = crawler_strategy
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def arun_many(self, *, urls, config, dispatcher=None):
        self.__class__.arun_many_calls.append(
            {
                "urls": list(urls),
                "config": config,
                "dispatcher": dispatcher,
                "strategy": self.crawler_strategy,
                "browser_config": self.config,
            }
        )

        if self.crawler_strategy is None:
            return [
                _FakeCrawlResult(
                    url="https://example.com/a",
                    success=True,
                    fit_markdown="A fit markdown",
                    raw_markdown="A raw markdown",
                    title="Article A",
                    metadata={"title": "Article A", "final_url": "https://example.com/a?ref=1"},
                    redirected_url="https://example.com/a?ref=1",
                ),
                _FakeCrawlResult(
                    url="https://example.com/b",
                    success=True,
                    fit_markdown="",
                    raw_markdown="B raw markdown",
                    title="Article B",
                    metadata={"title": "Article B"},
                ),
            ]

        return [
            _FakeCrawlResult(
                url="https://example.com/blocked",
                success=False,
                error_message="Blocked by anti-bot protection: DataDome captcha",
                status_code=403,
                title="Just a moment...",
                fit_markdown="",
                raw_markdown="",
                cleaned_html="",
                metadata={},
            )
        ]


def _install_fake_crawl4ai(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncWebCrawler.instances = []
    _FakeAsyncWebCrawler.arun_many_calls = []

    fake_module = ModuleType("crawl4ai")
    fake_module.AsyncWebCrawler = _FakeAsyncWebCrawler
    fake_module.BrowserConfig = _FakeBrowserConfig
    fake_module.CrawlerRunConfig = _FakeCrawlerRunConfig
    fake_module.CacheMode = SimpleNamespace(BYPASS="BYPASS")
    fake_module.DefaultMarkdownGenerator = _FakeDefaultMarkdownGenerator
    fake_module.PruningContentFilter = _FakePruningContentFilter
    fake_module.UndetectedAdapter = _FakeUndetectedAdapter

    fake_dispatcher_module = ModuleType("crawl4ai.async_dispatcher")
    fake_dispatcher_module.MemoryAdaptiveDispatcher = _FakeMemoryAdaptiveDispatcher

    fake_strategy_module = ModuleType("crawl4ai.async_crawler_strategy")
    fake_strategy_module.AsyncPlaywrightCrawlerStrategy = _FakeAsyncPlaywrightCrawlerStrategy

    monkeypatch.setitem(sys.modules, "crawl4ai", fake_module)
    monkeypatch.setitem(sys.modules, "crawl4ai.async_dispatcher", fake_dispatcher_module)
    monkeypatch.setitem(sys.modules, "crawl4ai.async_crawler_strategy", fake_strategy_module)


def test_crawl_urls_uses_shared_crawl4ai_session_and_prefers_fit_markdown(monkeypatch: pytest.MonkeyPatch):
    _install_fake_crawl4ai(monkeypatch)

    items, failures = crawl_urls(
        ["https://example.com/a", "https://example.com/b"],
        max_concurrency=2,
    )

    assert failures == []
    assert [item.url for item in items] == ["https://example.com/a", "https://example.com/b"]
    assert items[0].text == "A fit markdown"
    assert items[1].text == "B raw markdown"
    assert items[0].title == "Article A"
    assert items[0].metadata["final_url"] == "https://example.com/a?ref=1"
    assert items[0].origin_url == "https://example.com/a"
    assert len(_FakeAsyncWebCrawler.instances) == 1
    assert len(_FakeAsyncWebCrawler.arun_many_calls) == 1
    assert _FakeAsyncWebCrawler.arun_many_calls[0]["urls"] == ["https://example.com/a", "https://example.com/b"]
    assert _FakeAsyncWebCrawler.arun_many_calls[0]["dispatcher"].max_session_permit == 2


def test_crawl_urls_classifies_blocked_pages_stably(monkeypatch: pytest.MonkeyPatch):
    _install_fake_crawl4ai(monkeypatch)

    original_arun_many = _FakeAsyncWebCrawler.arun_many

    async def blocked_arun_many(self, *, urls, config, dispatcher=None):
        if self.crawler_strategy is None:
            return [
                _FakeCrawlResult(
                    url="https://www.reuters.com/story",
                    success=False,
                    error_message="Blocked by anti-bot protection: DataDome captcha",
                    status_code=403,
                    title="Just a moment...",
                    metadata={},
                )
            ]
        return await original_arun_many(self, urls=urls, config=config, dispatcher=dispatcher)

    monkeypatch.setattr(_FakeAsyncWebCrawler, "arun_many", blocked_arun_many)

    items, failures = crawl_urls(["https://www.reuters.com/story"], max_concurrency=1)

    assert items == []
    assert failures == [
        (
            "https://www.reuters.com/story",
            "blocked by anti-bot protection: DataDome captcha",
        )
    ]
    assert len(_FakeAsyncWebCrawler.instances) == 2
    assert _FakeAsyncWebCrawler.instances[1].crawler_strategy is not None


def test_extract_pdf_text_returns_empty_on_request_error(monkeypatch: pytest.MonkeyPatch):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers, follow_redirects):
            raise httpx.RequestError("network boom", request=httpx.Request("GET", url))

    monkeypatch.setattr("src.crawler.httpx.Client", lambda timeout: FakeClient())

    assert _extract_pdf_text("https://example.com/test.pdf") == ""


def test_crawl_pdf_url_reports_liteparse_install_hint_when_runtime_missing(monkeypatch: pytest.MonkeyPatch):
    pdf_bytes = b"%PDF-1.4 fake"

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers, follow_redirects):
            return SimpleNamespace(content=pdf_bytes, raise_for_status=lambda: None)

    class FakeCLINotFoundError(Exception):
        pass

    class FakeLiteParse:
        def parse(self, file_path, ocr_enabled=True):
            raise FakeCLINotFoundError("cli missing")

    fake_module = SimpleNamespace(LiteParse=FakeLiteParse, CLINotFoundError=FakeCLINotFoundError)

    monkeypatch.setattr("src.crawler.httpx.Client", lambda timeout: FakeClient())
    monkeypatch.setattr(
        "src.crawler.importlib.import_module",
        lambda name: fake_module if name == "liteparse" else __import__(name),
    )

    items, failures = crawl_urls(["https://example.com/report.pdf"])

    assert items == []
    assert failures == [
        (
            "https://example.com/report.pdf",
            "LiteParse is required for PDF parsing; install Node.js and run: npm install -g @llamaindex/liteparse",
        )
    ]


def test_crawl_pdf_url_uses_liteparse_text(monkeypatch: pytest.MonkeyPatch):
    pdf_bytes = b"%PDF-1.4 fake"

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers, follow_redirects):
            return SimpleNamespace(content=pdf_bytes, raise_for_status=lambda: None)

    class FakeLiteParse:
        def parse(self, file_path, ocr_enabled=True):
            assert ocr_enabled is True
            return SimpleNamespace(text="First page\n\nSecond page")

    fake_module = SimpleNamespace(LiteParse=FakeLiteParse, CLINotFoundError=RuntimeError)

    monkeypatch.setattr("src.crawler.httpx.Client", lambda timeout: FakeClient())
    monkeypatch.setattr(
        "src.crawler.importlib.import_module",
        lambda name: fake_module if name == "liteparse" else __import__(name),
    )

    items, failures = crawl_urls(["https://example.com/report.pdf"])

    assert failures == []
    assert len(items) == 1
    assert items[0].text == "First page\n\nSecond page"
    assert items[0].metadata == {"content_type": "application/pdf"}
    assert items[0].title == "report"

