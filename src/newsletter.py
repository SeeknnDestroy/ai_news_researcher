from __future__ import annotations

from .application.content_tasks import split_newsletter_items_async as _split_newsletter_items_async
from .domain.models import CrawlItem
from .infrastructure.llm_client import OpenAILLMClient
from .llm import OpenAIConfig


# Compatibility shim: keep the legacy module surface while the canonical
# implementation lives under src.application.content_tasks.
async def split_newsletter_items_async(
    config: OpenAIConfig,
    item: CrawlItem,
    max_items: int = 6,
) -> list[CrawlItem]:
    result = await _split_newsletter_items_async(
        OpenAILLMClient(config),
        item,
        max_items=max_items,
    )
    return result.items


__all__ = ["split_newsletter_items_async"]
