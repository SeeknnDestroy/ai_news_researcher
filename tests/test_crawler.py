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
    class FakeAsyncResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return FakeAsyncResponse("")

    monkeypatch.setattr("src.crawler.httpx.AsyncClient", FakeAsyncClient)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [("https://example.com/report", "Empty response from defuddle")]


def test_empty_defuddle_response_for_non_pdf_url_skips_pdf_fallback(monkeypatch):
    class FakeAsyncResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return FakeAsyncResponse("")

    def fail_if_called(url):
        raise AssertionError("PDF fallback should not run for non-PDF URLs")

    monkeypatch.setattr("src.crawler.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("src.crawler._crawl_pdf", fail_if_called)

    items, failures = crawl_urls(["https://example.com/report"])

    assert items == []
    assert failures == [("https://example.com/report", "Empty response from defuddle")]


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
