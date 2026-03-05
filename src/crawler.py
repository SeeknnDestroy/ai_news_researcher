from __future__ import annotations

import asyncio
import importlib
import io
import urllib.error
import urllib.request
from typing import List, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .config import CrawlItem


class CrawlError(RuntimeError):
    pass


def crawl_urls(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    """Return (items, failures)."""
    return asyncio.run(crawl_urls_async(urls, max_concurrency=max_concurrency))


async def crawl_urls_async(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    try:
        from crawl4ai import (
            AsyncWebCrawler,
            BrowserConfig,
            CrawlerRunConfig,
            DefaultMarkdownGenerator,
            PruningContentFilter,
        )
    except Exception as exc:  # pragma: no cover - dependency guard
        raise CrawlError(
            "crawl4ai is required. Install dependencies in requirements.txt"
        ) from exc

    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
        word_count_threshold=50,
        remove_forms=True,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        excluded_selector=(
            ".ads,.advert,.promo,.newsletter,.subscribe,.share,.social,"
            ".cookie,.cookies,.banner,.modal,.popup"
        ),
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(),
            options={
                "body_width": 0,
                "ignore_images": True,
                "ignore_links": False,
            },
        ),
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    items: List[CrawlItem] = []
    failures: List[tuple[str, str]] = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async def fetch(url: str):
            async with semaphore:
                try:
                    if _looks_like_pdf_url(url):
                        pdf_item = _crawl_pdf(url)
                        if pdf_item is not None:
                            items.append(pdf_item)
                            return

                    result = await crawler.arun(url=url, config=run_config)
                    if getattr(result, "success", True) is False:
                        failures.append((url, "crawl4ai returned success=False"))
                        return
                    item = _normalize_result(url, result)
                    if not item.text.strip():
                        # Fallback for binary sources or parser misses.
                        pdf_item = _crawl_pdf(url)
                        if pdf_item is not None:
                            items.append(pdf_item)
                            return
                    items.append(item)
                except Exception as exc:
                    failures.append((url, str(exc)))

        await asyncio.gather(*(fetch(url) for url in urls))

    return items, failures


def _normalize_result(url: str, result) -> CrawlItem:
    metadata = getattr(result, "metadata", None) or {}
    title = getattr(result, "title", None) or metadata.get("title")

    text = _select_markdown_text(result)

    if "<html" in text.lower():
        text = _html_to_text(text)

    return CrawlItem(url=url, text=text.strip(), metadata=metadata, title=title, origin_url=url)


def _select_markdown_text(result) -> str:
    markdown = getattr(result, "markdown", None)
    if markdown:
        for attr in ("fit_markdown", "markdown_with_citations", "raw_markdown"):
            value = getattr(markdown, attr, None)
            if value:
                return value
        if isinstance(markdown, str):
            return markdown

    return (
        getattr(result, "text", None)
        or getattr(result, "cleaned_html", None)
        or ""
    )


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(" ").split())


def _looks_like_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def _crawl_pdf(url: str) -> CrawlItem | None:
    text = _extract_pdf_text(url)
    if not text.strip():
        return None
    title = _title_from_pdf_url(url)
    return CrawlItem(url=url, text=text, metadata={"content_type": "application/pdf"}, title=title, origin_url=url)


def _extract_pdf_text(url: str) -> str:
    try:
        with httpx.Client(timeout=60) as client:
            headers = {"User-Agent": "ai-news-researcher/0.1", "Accept": "application/pdf,*/*"}
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            data = response.content
    except (httpx.RequestError, httpx.HTTPStatusError):
        return ""

    if not data:
        return ""

    try:
        pypdf_module = importlib.import_module("pypdf")
        PdfReader = getattr(pypdf_module, "PdfReader")
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""

    chunks = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            chunks.append(page_text.strip())

    return "\n\n".join(chunks).strip()


def _title_from_pdf_url(url: str) -> str:
    path = urlparse(url).path
    if not path:
        return "PDF Document"
    slug = path.rsplit("/", 1)[-1].strip() or "PDF Document"
    if slug.lower().endswith(".pdf"):
        slug = slug[:-4]
    slug = slug.replace("%20", " ").replace("-", " ").replace("_", " ").strip()
    return slug or "PDF Document"
