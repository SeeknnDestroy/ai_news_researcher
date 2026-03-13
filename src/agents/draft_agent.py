from __future__ import annotations

from typing import List

from ..application.report_tasks import generate_draft_outline as _generate_draft_outline
from ..domain.contracts import DraftOutline
from ..domain.models import SummaryItem
from ..infrastructure.llm_client import XAILLMClient
from ..llm import XAIConfig


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under src.application.report_tasks.
async def generate_draft_outline(
    config: XAIConfig,
    summaries: List[SummaryItem],
    critique: str = "",
    previous_draft: dict | DraftOutline | None = None,
) -> dict:
    outline = await _generate_draft_outline(
        XAILLMClient(config),
        summaries,
        critique=critique,
        previous_draft=DraftOutline.model_validate(previous_draft) if previous_draft else None,
    )
    return outline.model_dump(mode="json")
