from __future__ import annotations

import json
from typing import List

from ..config import SummaryItem, ExcludedItem
from ..llm import XAIConfig, generate_text_async
from ..templates.prompts import FINAL_REPORT_AGENT_SYSTEM_PROMPT, final_report_agent_user_prompt
from ..utils import log_stage

async def generate_final_report(
    config: XAIConfig,
    outline: dict,
    summaries: List[SummaryItem],
    excluded: List[ExcludedItem],
    critique: str = ""
) -> str:
    """
    Generates the final Markdown report based on the approved outline and original summaries.
    """
    log_stage("FINAL_AGENT", "Synthesizing final report")
    
    outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
    
    summaries_text = []
    for s in summaries:
        summaries_text.append(f"- URL: {s.url}")
        summaries_text.append(f"  Source: {s.source_name}")
        summaries_text.append(f"  Title: {s.title}")
        summaries_text.append(f"  Gelişme: {s.summary_tr}")
        summaries_text.append(f"  Neden Önemli: {s.why_it_matters_tr}\n")
    
    summaries_yaml = "\n".join(summaries_text)
    
    prompt = final_report_agent_user_prompt(
        outline_json=outline_str,
        summaries_yaml=summaries_yaml,
        critique=critique
    )
    
    report_markdown = await generate_text_async(
        config=config,
        system=FINAL_REPORT_AGENT_SYSTEM_PROMPT,
        user=prompt
    )
    
    if excluded:
        report_markdown += "\n\n## Kullanılamayan Kaynaklar\n"
        for ex in excluded:
            report_markdown += f"- {ex.url} — {ex.reason}\n"
            
    return report_markdown
