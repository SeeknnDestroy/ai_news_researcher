from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import os
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
                    from urllib.parse import quote

                    encoded_url = quote(url, safe='')
                    defuddle_url = f"https://defuddle.md/{encoded_url}"
                    response = await client.get(defuddle_url)
                    response.raise_for_status()

                    item = _parse_defuddle_response(url, response.text, yaml)
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
                    failures.append((url, f"HTTP {exc.response.status_code} from defuddle"))
                except Exception as exc:
                    failures.append((url, str(exc)))

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
