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
