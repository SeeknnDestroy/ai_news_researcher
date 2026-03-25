from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import json
import os
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
DEFUDDLE_HOSTED_BASE_URL = "https://defuddle.md"
DEFUDDLE_CLI_TIMEOUT_S = 90
DEFAULT_RETRY_AFTER_S = 1.0
MAX_HOSTED_DEFUDDLE_ATTEMPTS = 3


@dataclass(slots=True)
class PdfExtractionResult:
    text: str
    failure_reason: str | None = None


def crawl_urls(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    """Return (items, failures)."""
    return asyncio.run(crawl_urls_async(urls, max_concurrency=max_concurrency))


async def crawl_urls_async(urls: List[str], max_concurrency: int = 3) -> Tuple[List[CrawlItem], List[tuple[str, str]]]:
    import httpx
    import yaml

    semaphore = asyncio.Semaphore(max_concurrency)
    items: List[CrawlItem] = []
    failures: List[tuple[str, str]] = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        async def fetch(url: str):
            async with semaphore:
                try:
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

                    response_text = await _fetch_hosted_defuddle_text(client, url)
                    item = _parse_defuddle_response(url, response_text, yaml)
                    if not item.text.strip():
                        if _looks_like_pdf_url(url):
                            pdf_item, failure_reason = _crawl_pdf(url)
                            if pdf_item is not None:
                                items.append(pdf_item)
                                return
                            failures.append((url, failure_reason or "Empty response from defuddle and pdf fallback failed"))
                            return
                        failures.append((url, "Empty response from defuddle"))
                        return

                    items.append(item)
                except httpx.HTTPStatusError as exc:
                    hosted_reason = f"HTTP {exc.response.status_code} from defuddle"
                    local_reason = locals().get("local_error")
                    failures.append((url, _combine_defuddle_failures(local_reason, hosted_reason)))
                except Exception as exc:
                    hosted_reason = str(exc)
                    local_reason = locals().get("local_error")
                    failures.append((url, _combine_defuddle_failures(local_reason, hosted_reason)))

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
    return ["npx", "--yes", "defuddle"]


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


async def _fetch_hosted_defuddle_text(client: httpx.AsyncClient, url: str) -> str:
    from urllib.parse import quote

    encoded_url = quote(url, safe="")
    defuddle_url = f"{DEFUDDLE_HOSTED_BASE_URL}/{encoded_url}"

    for attempt_index in range(MAX_HOSTED_DEFUDDLE_ATTEMPTS):
        response = await client.get(defuddle_url)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            is_retryable = exc.response.status_code == 429
            has_attempts_remaining = attempt_index < (MAX_HOSTED_DEFUDDLE_ATTEMPTS - 1)
            if not is_retryable or not has_attempts_remaining:
                raise
            delay_seconds = _retry_after_seconds(exc.response.headers.get("retry-after"), attempt_index)
            await asyncio.sleep(delay_seconds)
            continue
        return response.text

    raise RuntimeError("Hosted defuddle retry loop exhausted unexpectedly")


def _retry_after_seconds(header_value: str | None, attempt_index: int) -> float:
    if not header_value:
        backoff_seconds = DEFAULT_RETRY_AFTER_S * (2**attempt_index)
        return backoff_seconds

    try:
        parsed_seconds = float(header_value)
    except ValueError:
        backoff_seconds = DEFAULT_RETRY_AFTER_S * (2**attempt_index)
        return backoff_seconds

    return max(0.0, parsed_seconds)


def _combine_defuddle_failures(local_reason: str | None, hosted_reason: str) -> str:
    if not local_reason:
        return hosted_reason
    return f"local defuddle failed ({local_reason}); hosted defuddle failed ({hosted_reason})"


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
