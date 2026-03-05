from __future__ import annotations

import json
from typing import List

from ..config import SummaryItem
from ..llm import XAIConfig, generate_json_async
from ..templates.prompts import DRAFT_AGENT_SYSTEM_PROMPT, draft_agent_user_prompt
from ..utils import log_stage

async def generate_draft_outline(config: XAIConfig, summaries: List[SummaryItem], critique: str = "", previous_draft: dict | None = None) -> dict:
    """
    Takes a list of SummaryItem objects and uses the Draft Agent to generate an outline.
    Order of summaries is arbitrary; the LLM handles prioritizing and sorting.
    """
    log_stage("DRAFT_AGENT", f"Generating outline from {len(summaries)} summaries")
    
    # Format summaries as a YAML-like string to feed into the prompt
    summaries_text = []
    for s in summaries:
        summaries_text.append(f"- URL: {s.url}")
        summaries_text.append(f"  Title: {s.title}")
        summaries_text.append(f"  Gelişme: {s.summary_tr}")
        summaries_text.append(f"  Neden Önemli: {s.why_it_matters_tr}\n")
    
    summaries_yaml = "\n".join(summaries_text)
    
    previous_draft_str = json.dumps(previous_draft, ensure_ascii=False, indent=2) if previous_draft else ""
    prompt = draft_agent_user_prompt(summaries_yaml, critique=critique, previous_draft=previous_draft_str)
    
    response = await generate_json_async(
        config=config,
        system=DRAFT_AGENT_SYSTEM_PROMPT,
        user=prompt
    )
    
    return response
