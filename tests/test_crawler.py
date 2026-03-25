import json
from types import SimpleNamespace

import httpx

from src.crawler import _extract_pdf_text, _parse_defuddle_response, crawl_urls
import yaml

def test_parse_defuddle_response():
    markdown = "---\ntitle: \"Moonshot lands a big one\"\nsource: \"https://example.com\"\n---\n\n# Main Content\n\nMore details."
    item = _parse_defuddle_response("https://example.com/article", markdown, yaml)
    
    assert item.title == "Moonshot lands a big one"
    assert item.url == "https://example.com/article"
    assert item.text == "# Main Content\n\nMore details."


def test_extract_pdf_text_returns_empty_on_request_error(monkeypatch):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers, follow_redirects):
            raise httpx.RequestError("network boom", request=httpx.Request("GET", url))

    monkeypatch.setattr("src.crawler.httpx.Client", lambda timeout: FakeClient())

    assert _extract_pdf_text("https://example.com/test.pdf") == ""


def test_crawl_pdf_url_reports_liteparse_install_hint_when_runtime_missing(monkeypatch):
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
    monkeypatch.setattr("src.crawler.importlib.import_module", lambda name: fake_module if name == "liteparse" else __import__(name))

    items, failures = crawl_urls(["https://example.com/report.pdf"])

    assert items == []
    assert failures == [
        (
            "https://example.com/report.pdf",
            "LiteParse is required for PDF parsing; install Node.js and run: npm install -g @llamaindex/liteparse",
        )
    ]


def test_empty_defuddle_response_for_non_pdf_url_returns_defuddle_failure(monkeypatch):
    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps({"content": "", "title": ""})
            self.stderr = ""

    async def fake_browser_fallback(url):
        del url
        return None, "Browser fallback returned empty content"

    async def fake_crawl4ai(url):
        del url
        return None, "crawl4ai returned empty content"

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fake_crawl4ai)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [
        (
            "https://example.com/report",
            "local defuddle failed (Empty response from defuddle); browser fallback failed (Browser fallback returned empty content); crawl4ai failed (crawl4ai returned empty content)",
        )
    ]


def test_empty_defuddle_response_for_non_pdf_url_skips_pdf_fallback(monkeypatch):
    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps({"content": "", "title": ""})
            self.stderr = ""

    def fail_if_called(url):
        raise AssertionError("PDF fallback should not run for non-PDF URLs")

    async def fake_browser_fallback(url):
        del url
        return None, "Browser fallback returned empty content"

    async def fake_crawl4ai(url):
        del url
        return None, "crawl4ai returned empty content"

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())
    monkeypatch.setattr("src.crawler._crawl_pdf", fail_if_called)
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fake_crawl4ai)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [
        (
            "https://example.com/report",
            "local defuddle failed (Empty response from defuddle); browser fallback failed (Browser fallback returned empty content); crawl4ai failed (crawl4ai returned empty content)",
        )
    ]


def test_crawl_pdf_url_uses_liteparse_text(monkeypatch):
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
    monkeypatch.setattr("src.crawler.importlib.import_module", lambda name: fake_module if name == "liteparse" else __import__(name))

    items, failures = crawl_urls(["https://example.com/report.pdf"])

    assert failures == []
    assert len(items) == 1
    assert items[0].text == "First page\n\nSecond page"
    assert items[0].metadata == {"content_type": "application/pdf"}
    assert items[0].title == "report"


def test_crawl_pdf_url_reports_stable_liteparse_failure(monkeypatch):
    pdf_bytes = b"%PDF-1.4 fake"

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers, follow_redirects):
            return SimpleNamespace(content=pdf_bytes, raise_for_status=lambda: None)

    class FakeParseError(Exception):
        pass

    class FakeLiteParse:
        def parse(self, file_path, ocr_enabled=True):
            raise FakeParseError("unexpected stderr")

    fake_module = SimpleNamespace(LiteParse=FakeLiteParse, CLINotFoundError=RuntimeError)

    monkeypatch.setattr("src.crawler.httpx.Client", lambda timeout: FakeClient())
    monkeypatch.setattr("src.crawler.importlib.import_module", lambda name: fake_module if name == "liteparse" else __import__(name))

    items, failures = crawl_urls(["https://example.com/report.pdf"])

    assert items == []
    assert failures == [("https://example.com/report.pdf", "LiteParse failed to parse PDF")]


def test_crawl_html_uses_local_defuddle_cli_before_hosted_api(monkeypatch):
    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "content": "# Main Content\n\nRecovered locally.",
                    "title": "Local Title",
                    "site": "Example Site",
                }
            )
            self.stderr = ""

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    items, failures = crawl_urls(["https://example.com/report"])

    assert failures == []
    assert len(items) == 1
    assert items[0].title == "Local Title"
    assert items[0].metadata["site"] == "Example Site"
    assert items[0].text == "# Main Content\n\nRecovered locally."


def test_crawl_html_falls_back_to_local_browser_when_local_cli_hits_fetch_failure(monkeypatch):
    class FakeCliProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = ""
            self.stderr = "Error: Failed to fetch: 403 Forbidden"

    browser_calls: list[str] = []

    async def fake_browser_fallback(url):
        browser_calls.append(url)
        return (
            SimpleNamespace(
                url=url,
                text="Browser body",
                metadata={"site": "Browser Site"},
                title="Browser Title",
                origin_url=url,
            ),
            None,
        )

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCliProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)

    items, failures = crawl_urls(["https://example.com/report"])

    assert failures == []
    assert len(items) == 1
    assert items[0].title == "Browser Title"
    assert items[0].metadata["site"] == "Browser Site"
    assert items[0].text == "Browser body"
    assert browser_calls == ["https://example.com/report"]


def test_crawl_html_does_not_use_browser_fallback_for_non_fetch_cli_failure(monkeypatch):
    class FakeCliProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = ""
            self.stderr = "Error: invalid json output"

    async def fail_if_called(url):
        raise AssertionError(f"browser fallback should not run for non-fetch failure: {url}")

    async def fail_crawl4ai_if_called(url):
        raise AssertionError(f"crawl4ai fallback should not run for non-fetch failure: {url}")

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCliProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fail_if_called)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fail_crawl4ai_if_called)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [("https://example.com/report", "invalid json output")]


def test_crawl_html_reports_combined_local_failures_when_browser_fallback_also_fails(monkeypatch):
    class FakeCliProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = ""
            self.stderr = "Error: Failed to fetch: 429 Too Many Requests"

    async def fake_browser_fallback(url):
        del url
        return None, "Playwright browser fallback failed"

    async def fake_crawl4ai_fallback(url):
        del url
        return None, "crawl4ai fallback failed"

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCliProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fake_crawl4ai_fallback)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [
        (
            "https://example.com/report",
            "local defuddle failed (Failed to fetch: 429 Too Many Requests); browser fallback failed (Playwright browser fallback failed); crawl4ai failed (crawl4ai fallback failed)",
        )
    ]


def test_crawl_html_falls_back_to_crawl4ai_when_browser_fallback_fails(monkeypatch):
    class FakeCliProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = ""
            self.stderr = "Error: Failed to fetch: 403 Forbidden"

    async def fake_browser_fallback(url):
        del url
        return None, "Blocked by anti-bot challenge"

    async def fake_crawl4ai_fallback(url):
        return (
            SimpleNamespace(
                url=url,
                text="Crawl4AI body",
                metadata={"site": "Crawl4AI Site"},
                title="Crawl4AI Title",
                origin_url=url,
            ),
            None,
        )

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCliProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fake_crawl4ai_fallback)

    items, failures = crawl_urls(["https://example.com/report"])

    assert failures == []
    assert len(items) == 1
    assert items[0].title == "Crawl4AI Title"
    assert items[0].metadata["site"] == "Crawl4AI Site"
    assert items[0].text == "Crawl4AI body"


def test_crawl_html_treats_anti_bot_challenge_as_failure(monkeypatch):
    class FakeCliProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "content": "Verification successful. Waiting for openai.com to respond",
                    "title": "Just a moment...",
                }
            )
            self.stderr = ""

    async def fake_browser_fallback(url):
        del url
        return None, "Blocked by anti-bot challenge"

    async def fake_crawl4ai_fallback(url):
        del url
        return None, "Blocked by anti-bot challenge"

    monkeypatch.setattr("src.crawler.subprocess.run", lambda *args, **kwargs: FakeCliProcess())
    monkeypatch.setattr("src.crawler._crawl_html_with_browser_defuddle", fake_browser_fallback)
    monkeypatch.setattr("src.crawler._crawl_html_with_crawl4ai", fake_crawl4ai_fallback)

    items, failures = crawl_urls(["https://openai.com/index/test"])

    assert items == []
    assert failures == [
        (
            "https://openai.com/index/test",
            "local defuddle failed (Blocked by anti-bot challenge); browser fallback failed (Blocked by anti-bot challenge); crawl4ai failed (Blocked by anti-bot challenge)",
        )
    ]
