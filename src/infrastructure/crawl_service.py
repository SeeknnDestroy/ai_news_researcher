from __future__ import annotations

from ..crawler import crawl_urls_async
from ..domain.models import CrawlFailure, CrawlStageResult


class Crawl4AICrawlService:
    async def crawl(self, urls: list[str], *, max_concurrency: int) -> CrawlStageResult:
        items, failures = await crawl_urls_async(urls, max_concurrency=max_concurrency)
        return CrawlStageResult(
            items=items,
            failures=[CrawlFailure(url=url, reason=reason) for url, reason in failures],
        )
