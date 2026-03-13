from __future__ import annotations

from typing import List

from ..application.report_tasks import generate_final_report as _generate_final_report
from ..domain.contracts import DraftOutline
from ..domain.models import ExcludedItem, SummaryItem
from ..infrastructure.llm_client import XAILLMClient
from ..llm import XAIConfig


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under src.application.report_tasks.
async def generate_final_report(
    config: XAIConfig,
    outline: dict,
    summaries: List[SummaryItem],
    excluded: List[ExcludedItem],
    critique: str = "",
) -> str:
    return await _generate_final_report(
        XAILLMClient(config),
        DraftOutline.model_validate(outline),
        summaries,
        excluded,
        critique=critique,
    )
