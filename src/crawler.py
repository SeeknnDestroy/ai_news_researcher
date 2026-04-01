from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, List, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .domain.models import CrawlItem


class CrawlError(RuntimeError):
    pass


LITEPARSE_INSTALL_HINT = (
    "LiteParse is required for PDF parsing; install Node.js and run: "
    "npm install -g @llamaindex/liteparse"
)
LITEPARSE_FAILURE_REASON = "LiteParse failed to parse PDF"

HTML_BLOCK_MARKERS = (
    "blocked by anti-bot protection",
    "anti-bot",
    "captcha",
    "cloudflare",
    "datadome",
    "just a moment",
    "enable javascript and cookies to continue",
    "verification successful",
    "forbidden",
)

HTML_EXCLUDED_TAGS = ["nav", "footer", "header", "aside", "script", "style"]
HTML_EXCLUDED_SELECTOR = (
    ".ads,.advert,.promo,.newsletter,.subscribe,.share,.social,"
    ".cookie,.cookies,.banner,.modal,.popup"
)


@dataclass(slots=True)
class PdfExtractionResult:
    text: str
    failure_reason: str | None = None


@dataclass(slots=True)
class Crawl4AIRuntime:
    AsyncWebCrawler: type
    BrowserConfig: type
    CacheMode: Any
    CrawlerRunConfig: type
    DefaultMarkdownGenerator: type
    PruningContentFilter: type
    MemoryAdaptiveDispatcher: type | None
    AsyncPlaywrightCrawlerStrategy: type | None
    UndetectedAdapter: type | None


def crawl_urls(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    """Return (items, failures)."""
    return asyncio.run(crawl_urls_async(urls, max_concurrency=max_concurrency))


async def crawl_urls_async(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    if not urls:
        return [], []

    html_urls = [url for url in urls if not _looks_like_pdf_url(url)]
    pdf_urls = [url for url in urls if _looks_like_pdf_url(url)]

    html_task = asyncio.create_task(_crawl_html_urls(html_urls, max_concurrency=max_concurrency)) if html_urls else None
    pdf_task = asyncio.create_task(_crawl_pdf_urls(pdf_urls)) if pdf_urls else None

    html_items_by_url: dict[str, CrawlItem] = {}
    html_failures_by_url: dict[str, str] = {}
    pdf_items_by_url: dict[str, CrawlItem] = {}
    pdf_failures_by_url: dict[str, str] = {}

    if html_task is not None:
        html_items_by_url, html_failures_by_url = await html_task

    if pdf_task is not None:
        pdf_items_by_url, pdf_failures_by_url = await pdf_task

    items: List[CrawlItem] = []
    failures: List[tuple[str, str]] = []

    for url in urls:
        if url in html_items_by_url:
            items.append(html_items_by_url[url])
            continue
        if url in pdf_items_by_url:
            items.append(pdf_items_by_url[url])
            continue
        if url in html_failures_by_url:
            failures.append((url, html_failures_by_url[url]))
            continue
        if url in pdf_failures_by_url:
            failures.append((url, pdf_failures_by_url[url]))

    return items, failures


async def _crawl_html_urls(urls: list[str], *, max_concurrency: int) -> tuple[dict[str, CrawlItem], dict[str, str]]:
    runtime = _load_crawl4ai_runtime()
    normal_outcomes = await _crawl_html_batch(
        runtime,
        urls,
        max_concurrency=max_concurrency,
        use_undetected=False,
    )
    items_by_url: dict[str, CrawlItem] = {}
    failures_by_url: dict[str, str] = {}
    blocked_urls: list[str] = []

    for url in urls:
        outcome = normal_outcomes.get(url)
        if outcome is None:
            failures_by_url[url] = "crawl4ai returned no result"
            continue
        if outcome.item is not None:
            items_by_url[url] = outcome.item
            continue
        if outcome.blocked:
            blocked_urls.append(url)
            continue
        failures_by_url[url] = outcome.failure_reason or "crawl4ai failed"

    if not blocked_urls:
        return items_by_url, failures_by_url

    retry_outcomes = await _crawl_html_batch(
        runtime,
        blocked_urls,
        max_concurrency=max(1, min(max_concurrency, len(blocked_urls))),
        use_undetected=True,
    )

    for url in blocked_urls:
        outcome = retry_outcomes.get(url)
        if outcome is None:
            failures_by_url[url] = "blocked by anti-bot protection"
            continue
        if outcome.item is not None:
            items_by_url[url] = outcome.item
            failures_by_url.pop(url, None)
            continue
        failures_by_url[url] = outcome.failure_reason or "blocked by anti-bot protection"

    return items_by_url, failures_by_url


async def _crawl_html_batch(
    runtime: Crawl4AIRuntime,
    urls: list[str],
    *,
    max_concurrency: int,
    use_undetected: bool,
) -> dict[str, "_CrawlOutcome"]:
    if not urls:
        return {}

    browser_config = _build_browser_config(runtime, use_undetected=use_undetected)
    run_config = _build_run_config(runtime)
    dispatcher = _build_dispatcher(runtime, max_concurrency)
    crawler_strategy = _build_crawler_strategy(runtime, browser_config, use_undetected=use_undetected)

    async with runtime.AsyncWebCrawler(config=browser_config, crawler_strategy=crawler_strategy) as crawler:
        raw_results = await _run_batch(crawler, urls, run_config, dispatcher)

    outcomes: dict[str, _CrawlOutcome] = {}
    for requested_url, result in zip(urls, raw_results):
        outcomes[requested_url] = _normalize_crawl4ai_result(requested_url, result)
    return outcomes


async def _run_batch(crawler, urls: list[str], run_config, dispatcher) -> list[object]:
    if hasattr(crawler, "arun_many"):
        results = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
        if hasattr(results, "__aiter__"):
            return [result async for result in results]
        return list(results)

    results: list[object] = []
    for url in urls:
        results.append(await crawler.arun(url=url, config=run_config))
    return results


def _build_browser_config(runtime: Crawl4AIRuntime, *, use_undetected: bool):
    return runtime.BrowserConfig(
        browser_type="chromium",
        headless=not use_undetected,
        verbose=False,
        enable_stealth=True,
    )


def _build_run_config(runtime: Crawl4AIRuntime):
    markdown_generator = runtime.DefaultMarkdownGenerator(
        content_filter=runtime.PruningContentFilter(),
    )
    return runtime.CrawlerRunConfig(
        cache_mode=runtime.CacheMode.BYPASS,
        remove_overlay_elements=True,
        remove_forms=True,
        excluded_tags=HTML_EXCLUDED_TAGS,
        excluded_selector=HTML_EXCLUDED_SELECTOR,
        markdown_generator=markdown_generator,
    )


def _build_dispatcher(runtime: Crawl4AIRuntime, max_concurrency: int):
    if runtime.MemoryAdaptiveDispatcher is None:
        return None
    return runtime.MemoryAdaptiveDispatcher(max_session_permit=max_concurrency, memory_threshold_percent=95.0)


def _build_crawler_strategy(runtime: Crawl4AIRuntime, browser_config, *, use_undetected: bool):
    if not use_undetected:
        return None
    if runtime.AsyncPlaywrightCrawlerStrategy is None or runtime.UndetectedAdapter is None:
        return None
    return runtime.AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_config,
        browser_adapter=runtime.UndetectedAdapter(),
    )


@dataclass(slots=True)
class _CrawlOutcome:
    item: CrawlItem | None
    failure_reason: str | None
    blocked: bool = False


def _normalize_crawl4ai_result(requested_url: str, result: object | None) -> _CrawlOutcome:
    if result is None:
        return _CrawlOutcome(item=None, failure_reason="crawl4ai returned no result", blocked=False)

    success = bool(getattr(result, "success", False))
    error_message = str(getattr(result, "error_message", "") or "").strip()
    status_code = getattr(result, "status_code", None)
    metadata = _metadata_from_result(result)
    final_url = _final_url_from_result(result, requested_url)
    title = _title_from_result(result, metadata)
    markdown_text = _best_markdown_text(result)

    if final_url and final_url != requested_url:
        metadata.setdefault("final_url", final_url)

    if success:
        if _looks_like_block_page_text(title, markdown_text, error_message):
            return _CrawlOutcome(
                item=None,
                failure_reason=_blocked_failure_reason(result, error_message=error_message, status_code=status_code),
                blocked=True,
            )
        if not markdown_text.strip():
            if _looks_like_block_page_text(title, markdown_text, error_message):
                return _CrawlOutcome(
                    item=None,
                    failure_reason=_blocked_failure_reason(result, error_message=error_message, status_code=status_code),
                    blocked=True,
                )
            return _CrawlOutcome(item=None, failure_reason="crawl4ai returned empty content", blocked=False)
        return _CrawlOutcome(
            item=CrawlItem(
                url=requested_url,
                text=markdown_text.strip(),
                metadata=metadata,
                title=title,
                origin_url=requested_url,
            ),
            failure_reason=None,
            blocked=False,
        )

    blocked = _looks_like_block_page_text(title, markdown_text, error_message)
    if status_code in {401, 403, 429}:
        blocked = True
    if blocked:
        return _CrawlOutcome(
            item=None,
            failure_reason=_blocked_failure_reason(result, error_message=error_message, status_code=status_code),
            blocked=True,
        )

    failure_reason = error_message or f"crawl4ai failed{f' ({status_code})' if status_code else ''}"
    return _CrawlOutcome(item=None, failure_reason=failure_reason, blocked=False)


def _best_markdown_text(result: object) -> str:
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        cleaned_html = str(getattr(result, "cleaned_html", "") or "").strip()
        if cleaned_html:
            return _html_to_text(cleaned_html)
        return str(getattr(result, "text", "") or "").strip()

    for attr_name in ("fit_markdown", "raw_markdown", "markdown"):
        value = getattr(markdown, attr_name, None)
        if value:
            return str(value).strip()

    cleaned_html = str(getattr(result, "cleaned_html", "") or "").strip()
    if cleaned_html:
        return _html_to_text(cleaned_html)

    return str(getattr(result, "text", "") or "").strip()


def _metadata_from_result(result: object) -> dict[str, object]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _final_url_from_result(result: object, requested_url: str) -> str:
    redirected_url = str(getattr(result, "redirected_url", "") or "").strip()
    if redirected_url:
        return redirected_url
    result_url = str(getattr(result, "url", "") or "").strip()
    if result_url:
        return result_url
    return requested_url


def _title_from_result(result: object, metadata: dict[str, object]) -> str | None:
    title_value = metadata.get("title")
    if title_value:
        return str(title_value).strip() or None
    result_title = getattr(result, "title", None)
    if result_title:
        return str(result_title).strip() or None
    return None


def _blocked_failure_reason(result: object, *, error_message: str, status_code: object) -> str:
    normalized = f"{error_message} {status_code or ''}".lower()
    if "datadome" in normalized and "captcha" in normalized:
        return "blocked by anti-bot protection: DataDome captcha"
    if "cloudflare" in normalized:
        return "blocked by anti-bot protection: Cloudflare challenge"
    if "captcha" in normalized:
        return "blocked by anti-bot protection: captcha"
    if "forbidden" in normalized or "403" in normalized:
        return "blocked by anti-bot protection: 403 forbidden"
    return "blocked by anti-bot protection"


def _looks_like_block_page_text(title: str | None, text: str, error_message: str) -> bool:
    combined_text = f"{title or ''}\n{text}\n{error_message}".lower()
    return any(marker in combined_text for marker in HTML_BLOCK_MARKERS)


async def _crawl_pdf_urls(urls: list[str]) -> tuple[dict[str, CrawlItem], dict[str, str]]:
    items: dict[str, CrawlItem] = {}
    failures: dict[str, str] = {}

    for url in urls:
        pdf_item, failure_reason = await asyncio.to_thread(_crawl_pdf, url)
        if pdf_item is not None:
            items[url] = pdf_item
            continue
        failures[url] = failure_reason or "Failed to extract text from PDF"

    return items, failures


def _crawl_pdf(url: str) -> tuple[CrawlItem | None, str | None]:
    extraction = _extract_pdf(url)
    if not extraction.text.strip():
        return None, extraction.failure_reason or "Failed to extract text from PDF"
    title = _title_from_pdf_url(url)
    return (
        CrawlItem(
            url=url,
            text=extraction.text,
            metadata={"content_type": "application/pdf"},
            title=title,
            origin_url=url,
        ),
        None,
    )


def _extract_pdf(url: str) -> PdfExtractionResult:
    text = _extract_pdf_text(url)
    if text.strip():
        return PdfExtractionResult(text=text)

    failure_reason = getattr(_extract_pdf_text, "_last_failure_reason", None)
    if isinstance(failure_reason, str) and failure_reason:
        return PdfExtractionResult(text="", failure_reason=failure_reason)
    return PdfExtractionResult(text="", failure_reason=LITEPARSE_FAILURE_REASON)


def _extract_pdf_text(url: str) -> str:
    try:
        with httpx.Client(timeout=60) as client:
            headers = {"User-Agent": "ai-news-researcher/0.1", "Accept": "application/pdf,*/*"}
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            data = response.content
    except (httpx.RequestError, httpx.HTTPStatusError):
        _extract_pdf_text._last_failure_reason = ""
        return ""

    if not data:
        _extract_pdf_text._last_failure_reason = LITEPARSE_FAILURE_REASON
        return ""

    try:
        liteparse_module = importlib.import_module("liteparse")
        LiteParse = getattr(liteparse_module, "LiteParse")
        cli_not_found_error = getattr(liteparse_module, "CLINotFoundError", RuntimeError)
    except Exception:
        _extract_pdf_text._last_failure_reason = LITEPARSE_INSTALL_HINT
        return ""

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        try:
            handle.write(data)
            temp_path = Path(handle.name)
        finally:
            handle.flush()

    try:
        parser = LiteParse()
        parse_result = parser.parse(str(temp_path), ocr_enabled=True)
        parsed_text = str(getattr(parse_result, "text", "") or "").strip()
        if parsed_text:
            _extract_pdf_text._last_failure_reason = None
            return parsed_text
        _extract_pdf_text._last_failure_reason = LITEPARSE_FAILURE_REASON
        return ""
    except cli_not_found_error:
        _extract_pdf_text._last_failure_reason = LITEPARSE_INSTALL_HINT
        return ""
    except Exception:
        _extract_pdf_text._last_failure_reason = LITEPARSE_FAILURE_REASON
        return ""
    finally:
        temp_path.unlink(missing_ok=True)


def _title_from_pdf_url(url: str) -> str:
    path = urlparse(url).path
    if not path:
        return "PDF Document"
    slug = path.rsplit("/", 1)[-1].strip() or "PDF Document"
    if slug.lower().endswith(".pdf"):
        slug = slug[:-4]
    slug = slug.replace("%20", " ").replace("-", " ").replace("_", " ").strip()
    return slug or "PDF Document"


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(" ").split())


def _looks_like_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def _load_crawl4ai_runtime() -> Crawl4AIRuntime:
    try:
        crawl4ai_module = importlib.import_module("crawl4ai")
    except ImportError as exc:
        raise CrawlError("crawl4ai is required. Install project dependencies from pyproject.toml.") from exc

    AsyncWebCrawler = getattr(crawl4ai_module, "AsyncWebCrawler", None)
    BrowserConfig = getattr(crawl4ai_module, "BrowserConfig", None)
    CacheMode = getattr(crawl4ai_module, "CacheMode", None)
    CrawlerRunConfig = getattr(crawl4ai_module, "CrawlerRunConfig", None)
    DefaultMarkdownGenerator = getattr(crawl4ai_module, "DefaultMarkdownGenerator", None)
    PruningContentFilter = getattr(crawl4ai_module, "PruningContentFilter", None)
    UndetectedAdapter = getattr(crawl4ai_module, "UndetectedAdapter", None)

    if any(
        value is None
        for value in (
            AsyncWebCrawler,
            BrowserConfig,
            CacheMode,
            CrawlerRunConfig,
            DefaultMarkdownGenerator,
            PruningContentFilter,
        )
    ):
        raise CrawlError("crawl4ai is missing required runtime classes.")

    MemoryAdaptiveDispatcher = None
    try:
        dispatcher_module = importlib.import_module("crawl4ai.async_dispatcher")
        MemoryAdaptiveDispatcher = getattr(dispatcher_module, "MemoryAdaptiveDispatcher", None)
    except Exception:
        MemoryAdaptiveDispatcher = None

    AsyncPlaywrightCrawlerStrategy = None
    try:
        strategy_module = importlib.import_module("crawl4ai.async_crawler_strategy")
        AsyncPlaywrightCrawlerStrategy = getattr(strategy_module, "AsyncPlaywrightCrawlerStrategy", None)
    except Exception:
        AsyncPlaywrightCrawlerStrategy = None

    return Crawl4AIRuntime(
        AsyncWebCrawler=AsyncWebCrawler,
        BrowserConfig=BrowserConfig,
        CacheMode=CacheMode,
        CrawlerRunConfig=CrawlerRunConfig,
        DefaultMarkdownGenerator=DefaultMarkdownGenerator,
        PruningContentFilter=PruningContentFilter,
        MemoryAdaptiveDispatcher=MemoryAdaptiveDispatcher,
        AsyncPlaywrightCrawlerStrategy=AsyncPlaywrightCrawlerStrategy,
        UndetectedAdapter=UndetectedAdapter,
    )
