from __future__ import annotations

from .application.ai_tasks import split_newsletter_items_async as _split_newsletter_items_async
from .config import CrawlItem
from .infrastructure.llm_client import XAILLMClient
from .llm import XAIConfig


async def split_newsletter_items_async(
    config: XAIConfig,
    item: CrawlItem,
    max_items: int = 6,
) -> list[CrawlItem]:
    result = await _split_newsletter_items_async(
        XAILLMClient(config),
        item,
        max_items=max_items,
    )
    return result.items


__all__ = ["split_newsletter_items_async"]
