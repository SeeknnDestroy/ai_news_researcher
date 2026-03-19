from __future__ import annotations

import asyncio
import importlib
import io
import httpx
from typing import List, Tuple
from urllib.parse import urlparse



from .domain.models import CrawlItem


class CrawlError(RuntimeError):
    pass


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
                        pdf_item = _crawl_pdf(url)
                        if pdf_item is not None:
                            items.append(pdf_item)
                            return
                        # If pdf extraction yields nothing, let's treat it as failure
                        failures.append((url, "Failed to extract text from PDF"))
                        return
                    from urllib.parse import quote
                    
                    encoded_url = quote(url, safe='')
                    defuddle_url = f"https://defuddle.md/{encoded_url}"
                    response = await client.get(defuddle_url)
                    response.raise_for_status()

                    item = _parse_defuddle_response(url, response.text, yaml)
                    if not item.text.strip():
                        # Fallback for binary sources or parser misses.
                        pdf_item = _crawl_pdf(url)
                        if pdf_item is not None:
                            items.append(pdf_item)
                            return
                        failures.append((url, "Empty response from defuddle and pdf fallback failed"))
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
