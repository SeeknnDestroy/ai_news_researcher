from __future__ import annotations

from .application.ai_tasks import summarize_article_async as _summarize_article_async
from .config import CrawlItem, SummaryItem
from .infrastructure.llm_client import XAILLMClient
from .llm import XAIConfig


async def summarize_article_async(config: XAIConfig, item: CrawlItem, audience: str = "mixed") -> SummaryItem:
    return await _summarize_article_async(XAILLMClient(config), item, audience=audience)


__all__ = ["summarize_article_async"]
