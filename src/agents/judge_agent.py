from __future__ import annotations

from ..application.report_tasks import evaluate_draft_outline as _evaluate_draft_outline
from ..domain.contracts import DraftOutline
from ..infrastructure.llm_client import XAILLMClient
from ..llm import XAIConfig


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under src.application.report_tasks.
async def evaluate_draft_outline(config: XAIConfig, draft_outline: dict, previous_critiques: str = "") -> dict:
    evaluation = await _evaluate_draft_outline(
        XAILLMClient(config),
        DraftOutline.model_validate(draft_outline),
        previous_critiques=previous_critiques,
    )
    return evaluation.model_dump(mode="json")
