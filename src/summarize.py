from __future__ import annotations

from .application.content_tasks import summarize_article_async as _summarize_article_async
from .domain.models import CrawlItem, SummaryItem
from .infrastructure.llm_client import OpenAILLMClient
from .llm import OpenAIConfig


# Compatibility shim: keep the legacy module surface while the canonical
# implementation lives under src.application.content_tasks.
async def summarize_article_async(config: OpenAIConfig, item: CrawlItem, audience: str = "mixed") -> SummaryItem:
    return await _summarize_article_async(OpenAILLMClient(config), item, audience=audience)


__all__ = ["summarize_article_async"]
