from __future__ import annotations

from typing import List

from ..application.ai_tasks import generate_final_report as _generate_final_report
from ..config import ExcludedItem, SummaryItem
from ..domain.contracts import DraftOutline
from ..infrastructure.llm_client import XAILLMClient
from ..llm import XAIConfig

async def generate_final_report(
    config: XAIConfig,
    outline: dict,
    summaries: List[SummaryItem],
    excluded: List[ExcludedItem],
    critique: str = ""
) -> str:
    return await _generate_final_report(
        XAILLMClient(config),
        DraftOutline.model_validate(outline),
        summaries,
        excluded,
        critique=critique,
    )
