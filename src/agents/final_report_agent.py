from __future__ import annotations

import json
from typing import List

import asyncio
from ..config import SummaryItem, ExcludedItem
from ..llm import XAIConfig, generate_text_async
from ..templates.prompts import THEME_REPORT_AGENT_SYSTEM_PROMPT, theme_report_agent_user_prompt
from ..utils import log_stage, format_date

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
    
    report_parts = []
    report_title = outline.get("report_title", "GenAI Haftalık Rapor")
    intro = outline.get("introduction_commentary", "")
    
    if intro:
        report_parts.append(f"# {report_title}\n\n{intro}\n")
    else:
        report_parts.append(f"# {report_title}\n")
        
    tasks = []
    for theme in outline.get("themes", []):
        theme_json = json.dumps(theme, ensure_ascii=False, indent=2)
        
        theme_urls = set()
        for article in theme.get("articles", []):
            for u in article.get("news_urls_included", []):
                theme_urls.add(u)
                
        theme_summaries = [s for s in summaries if s.url in theme_urls]
        if not theme_summaries:
            theme_summaries = summaries
            
        summaries_text = []
        for s in theme_summaries:
            summaries_text.append(f"- URL: {s.url}")
            summaries_text.append(f"  Source: {s.source_name}")
            summaries_text.append(f"  Date: {format_date(s.date)}")
            summaries_text.append(f"  Title: {s.title}")
            summaries_text.append(f"  Gelişme: {s.summary_tr}")
            summaries_text.append(f"  Neden Önemli: {s.why_it_matters_tr}\n")
            
        summaries_yaml = "\n".join(summaries_text)
        
        prompt = theme_report_agent_user_prompt(
            theme_json=theme_json,
            summaries_yaml=summaries_yaml,
            critique=critique
        )
        
        tasks.append(generate_text_async(
            config=config,
            system=THEME_REPORT_AGENT_SYSTEM_PROMPT,
            user=prompt
        ))
        
    theme_results = await asyncio.gather(*tasks)
    
    for res in theme_results:
        report_parts.append(res)
        report_parts.append("\n")
        
    report_markdown = "\n".join(report_parts)
    
    if excluded:
        report_markdown += "\n\n## Kullanılamayan Kaynaklar\n"
        for ex in excluded:
            report_markdown += f"- {ex.url} — {ex.reason}\n"
            
    return report_markdown
