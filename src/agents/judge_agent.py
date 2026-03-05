from __future__ import annotations

import json

from ..llm import XAIConfig, generate_json_async
from ..templates.prompts import JUDGE_AGENT_SYSTEM_PROMPT, judge_agent_user_prompt
from ..utils import log_stage

async def evaluate_draft_outline(config: XAIConfig, draft_outline: dict) -> dict:
    """
    Evaluates the draft outline based on heuristics (no BS headings, professionalism, avoiding 'banka' spam).
    Returns a dict with `passes_criteria`, `critique`, and `specific_fixes_required`.
    """
    log_stage("JUDGE_AGENT", "Evaluating draft outline")
    
    draft_json_str = json.dumps(draft_outline, ensure_ascii=False, indent=2)
    prompt = judge_agent_user_prompt(draft_json_str)
    
    response = await generate_json_async(
        config=config,
        system=JUDGE_AGENT_SYSTEM_PROMPT,
        user=prompt
    )
    
    return response
