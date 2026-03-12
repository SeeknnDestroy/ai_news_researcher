from types import SimpleNamespace

import httpx

from src.crawler import _extract_pdf_text, _select_markdown_text, crawl_urls


def test_select_markdown_text_prefers_fit_markdown():
    markdown = SimpleNamespace(
        fit_markdown="fit",
        markdown_with_citations="citations",
        raw_markdown="raw",
    )
    result = SimpleNamespace(markdown=markdown)

    assert _select_markdown_text(result) == "fit"


def test_crawl_raw_html_filters_chrome_and_keeps_headings():
    html = (
        "<html><body>"
        "<header><nav>Explore Courses</nav></header>"
        "<article>"
        "<h1>Moonshot lands a big one</h1>"
        "<p>Main point of the article.</p>"
        "<h2>Key details</h2>"
        "<p>More details here.</p>"
        "</article>"
        "<footer>Subscribe now</footer>"
        "</body></html>"
    )

    items, failures = crawl_urls([f"raw:{html}"], max_concurrency=1)

    assert failures == []
    assert len(items) == 1

    text = items[0].text
    assert "# Moonshot lands a big one" in text
    assert "## Key details" in text
    assert "Explore Courses" not in text
    assert "Subscribe now" not in text


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
