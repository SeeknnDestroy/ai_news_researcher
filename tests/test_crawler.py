from types import SimpleNamespace

from src.crawler import _select_markdown_text, crawl_urls


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
