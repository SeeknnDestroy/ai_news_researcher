from __future__ import annotations

from ..application.ai_tasks import evaluate_draft_outline as _evaluate_draft_outline
from ..domain.contracts import DraftOutline
from ..infrastructure.llm_client import XAILLMClient
from ..llm import XAIConfig


async def evaluate_draft_outline(config: XAIConfig, draft_outline: dict, previous_critiques: str = "") -> dict:
    evaluation = await _evaluate_draft_outline(
        XAILLMClient(config),
        DraftOutline.model_validate(draft_outline),
        previous_critiques=previous_critiques,
    )
    return evaluation.model_dump(mode="json")
