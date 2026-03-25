from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import httpx
from typing import List, Tuple
from urllib.parse import urlparse

from .domain.models import CrawlItem


class CrawlError(RuntimeError):
    pass


LITEPARSE_INSTALL_HINT = (
    "LiteParse is required for PDF parsing; install Node.js and run: "
    "npm install -g @llamaindex/liteparse"
)
LITEPARSE_FAILURE_REASON = "LiteParse failed to parse PDF"
DEFUDDLE_CLI_TIMEOUT_S = 90
BROWSER_FALLBACK_TIMEOUT_S = 90


@dataclass(slots=True)
class PdfExtractionResult:
    text: str
    failure_reason: str | None = None


def crawl_urls(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    """Return (items, failures)."""
    return asyncio.run(crawl_urls_async(urls, max_concurrency=max_concurrency))


async def crawl_urls_async(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    items: List[CrawlItem] = []
    failures: List[tuple[str, str]] = []

    async def fetch(url: str):
        async with semaphore:
            if _looks_like_pdf_url(url):
                pdf_item, failure_reason = _crawl_pdf(url)
                if pdf_item is not None:
                    items.append(pdf_item)
                    return
                failures.append((url, failure_reason or "Failed to extract text from PDF"))
                return

            item, local_error = await _crawl_html_with_local_defuddle(url)
            if item is not None:
                items.append(item)
                return

            if not local_error:
                failures.append((url, "Empty response from defuddle"))
                return

            if not _should_attempt_browser_fallback(local_error):
                failures.append((url, local_error))
                return

            browser_item, browser_error = await _crawl_html_with_browser_defuddle(url)
            if browser_item is not None:
                items.append(browser_item)
                return

            crawl4ai_item, crawl4ai_error = await _crawl_html_with_crawl4ai(url)
            if crawl4ai_item is not None:
                items.append(crawl4ai_item)
                return

            failures.append((url, _combine_local_failures(local_error, browser_error, crawl4ai_error)))

    await asyncio.gather(*(fetch(url) for url in urls))

    return items, failures


def _parse_defuddle_response(url: str, text: str, yaml_module) -> CrawlItem:
    text = text.strip()
    title = None
    metadata = {}
    
    # Parse YAML frontmatter if present
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml_module.safe_load(parts[1])
                if isinstance(frontmatter, dict):
                    metadata = frontmatter
                    title = metadata.get("title")
            except Exception:
                pass
            text = parts[2].strip()

    return CrawlItem(url=url, text=text, metadata=metadata, title=title, origin_url=url)


async def _crawl_html_with_local_defuddle(url: str) -> tuple[CrawlItem | None, str | None]:
    try:
        item = await asyncio.to_thread(_run_local_defuddle_cli, url)
    except Exception as exc:
        return None, str(exc)
    if not item.text.strip():
        return None, "Empty response from defuddle"
    if _looks_like_block_page(item):
        return None, "Blocked by anti-bot challenge"
    return item, None


def _run_local_defuddle_cli(url: str) -> CrawlItem:
    command = _defuddle_command()
    command.extend(["parse", url, "--json", "--markdown"])
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=DEFUDDLE_CLI_TIMEOUT_S,
    )

    if process.returncode != 0:
        failure_message = _defuddle_process_error(process)
        raise RuntimeError(failure_message)

    payload = json.loads(process.stdout)
    return _parse_defuddle_json_payload(url, payload)


def _defuddle_command() -> list[str]:
    binary_path = shutil.which("defuddle")
    if binary_path:
        return [binary_path]
    return ["./node_modules/.bin/defuddle"] if Path("node_modules/.bin/defuddle").exists() else ["npx", "--yes", "defuddle"]


def _defuddle_process_error(process: subprocess.CompletedProcess[str]) -> str:
    stderr_text = (process.stderr or "").strip()
    stdout_text = (process.stdout or "").strip()
    raw_message = stderr_text or stdout_text or "defuddle CLI failed"
    cleaned_message = raw_message.replace("Error: ", "").strip()
    return cleaned_message


def _parse_defuddle_json_payload(url: str, payload: dict[str, object]) -> CrawlItem:
    metadata = dict(payload)
    title_value = metadata.get("title")
    content_value = metadata.get("content")
    title_text = str(title_value).strip() if title_value else None
    content_text = str(content_value or "").strip()
    return CrawlItem(
        url=url,
        text=content_text,
        metadata=metadata,
        title=title_text,
        origin_url=url,
    )


async def _crawl_html_with_browser_defuddle(url: str) -> tuple[CrawlItem | None, str | None]:
    try:
        item = await asyncio.to_thread(_run_browser_defuddle_helper, url)
    except Exception as exc:
        return None, str(exc)
    if not item.text.strip():
        return None, "Browser fallback returned empty content"
    if _looks_like_block_page(item):
        return None, "Blocked by anti-bot challenge"
    return item, None


async def _crawl_html_with_crawl4ai(url: str) -> tuple[CrawlItem | None, str | None]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    except Exception:
        return None, "crawl4ai is not installed"

    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url, config=run_config)
    except Exception as exc:
        return None, str(exc)

    if not result.success:
        error_message = str(getattr(result, "error_message", "") or "crawl4ai failed")
        return None, error_message

    markdown_value = getattr(result, "markdown", "") or ""
    title_value = None
    metadata_value = getattr(result, "metadata", None)
    if isinstance(metadata_value, dict):
        raw_title = metadata_value.get("title")
        title_value = str(raw_title).strip() if raw_title else None

    item = CrawlItem(
        url=str(getattr(result, "url", "") or url),
        text=str(markdown_value).strip(),
        metadata=dict(metadata_value) if isinstance(metadata_value, dict) else {},
        title=title_value,
        origin_url=url,
    )

    if not item.text.strip():
        return None, "crawl4ai returned empty content"
    if _looks_like_block_page(item):
        return None, "Blocked by anti-bot challenge"
    return item, None


def _run_browser_defuddle_helper(url: str) -> CrawlItem:
    helper_path = Path(__file__).resolve().parent / "browser_fetch_and_parse.mjs"
    process = subprocess.run(
        ["node", str(helper_path), url],
        capture_output=True,
        text=True,
        timeout=BROWSER_FALLBACK_TIMEOUT_S,
    )

    if process.returncode != 0:
        failure_message = _defuddle_process_error(process)
        raise RuntimeError(failure_message)

    payload = json.loads(process.stdout)
    metadata = payload.get("metadata")
    final_url = str(payload.get("final_url") or url)
    title = str(payload.get("title") or "").strip() or None
    content = str(payload.get("content") or "").strip()
    return CrawlItem(
        url=final_url,
        text=content,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        title=title,
        origin_url=url,
    )


def _should_attempt_browser_fallback(local_error: str) -> bool:
    normalized_error = local_error.lower()
    fetch_markers = (
        "failed to fetch",
        "too many requests",
        "forbidden",
        "timed out",
        "timeout",
        "empty response",
        "no content could be extracted",
        "net::",
        "network",
        "dns",
        "anti-bot challenge",
    )
    return any(marker in normalized_error for marker in fetch_markers)


def _combine_local_failures(local_reason: str, browser_reason: str | None, crawl4ai_reason: str | None) -> str:
    if not browser_reason and not crawl4ai_reason:
        return local_reason
    message = f"local defuddle failed ({local_reason})"
    if browser_reason:
        message += f"; browser fallback failed ({browser_reason})"
    if crawl4ai_reason:
        message += f"; crawl4ai failed ({crawl4ai_reason})"
    return message


def _looks_like_block_page(item: CrawlItem) -> bool:
    title_text = (item.title or "").strip().lower()
    content_sample = item.text[:400].strip().lower()
    challenge_markers = (
        "just a moment",
        "enable javascript and cookies to continue",
        "verification successful. waiting for",
        "cf challenge",
        "cloudflare",
    )
    combined_text = f"{title_text}\n{content_sample}"
    return any(marker in combined_text for marker in challenge_markers)


def _looks_like_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


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


def _extract_pdf_text(url: str) -> str:
    return _extract_pdf(url).text


def _extract_pdf(url: str) -> PdfExtractionResult:
    try:
        with httpx.Client(timeout=60) as client:
            headers = {"User-Agent": "ai-news-researcher/0.1", "Accept": "application/pdf,*/*"}
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            data = response.content
    except (httpx.RequestError, httpx.HTTPStatusError):
        return PdfExtractionResult(text="")

    if not data:
        return PdfExtractionResult(text="")

    return _extract_pdf_text_from_bytes(data)


def _extract_pdf_text_from_bytes(data: bytes) -> PdfExtractionResult:
    cli_not_found_error: type[BaseException] | tuple[type[BaseException], ...] = ()

    try:
        liteparse_module = importlib.import_module("liteparse")
        LiteParse = getattr(liteparse_module, "LiteParse")
        maybe_cli_not_found = getattr(liteparse_module, "CLINotFoundError", None)
        if isinstance(maybe_cli_not_found, type) and issubclass(maybe_cli_not_found, BaseException):
            cli_not_found_error = maybe_cli_not_found
    except Exception:
        return PdfExtractionResult(text="", failure_reason="LiteParse Python package is not installed")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        parser = LiteParse()
        result = parser.parse(temp_path, ocr_enabled=True)
    except cli_not_found_error:
        return PdfExtractionResult(text="", failure_reason=LITEPARSE_INSTALL_HINT)
    except Exception:
        return PdfExtractionResult(text="", failure_reason=LITEPARSE_FAILURE_REASON)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    text = (getattr(result, "text", "") or "").strip()
    if not text:
        return PdfExtractionResult(text="", failure_reason="LiteParse returned no text from PDF")

    return PdfExtractionResult(text=text)


def _title_from_pdf_url(url: str) -> str:
    path = urlparse(url).path
    if not path:
        return "PDF Document"
    slug = path.rsplit("/", 1)[-1].strip() or "PDF Document"
    if slug.lower().endswith(".pdf"):
        slug = slug[:-4]
    slug = slug.replace("%20", " ").replace("-", " ").replace("_", " ").strip()
    return slug or "PDF Document"
