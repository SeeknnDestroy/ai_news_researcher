from __future__ import annotations

from typing import List

from .config import SummaryItem, ThemeGroup
from .llm import XAIConfig, LLMError, generate_json


SYSTEM_PROMPT = """
You are a strict technical editor for a weekly GenAI report used by a bank technology team.
Group items into coherent Turkish themes for fast executive/engineering scanning.
Return ONLY valid JSON in the requested schema.
""".strip()


def group_themes(config: XAIConfig, items: List[SummaryItem]) -> List[ThemeGroup]:
    if not items:
        return []

    payload_lines = []
    for idx, item in enumerate(items):
        payload_lines.append(
            f"{idx}. {item.title} | tags: {', '.join(item.tags)} | summary: {item.summary_tr}"
        )

    user_prompt = """
Aşağıdaki haberleri temalara ayır.
JSON çıktısı zorunlu:
{
  "themes": [
    {"name": "Theme name in Turkish", "item_ids": [0,1]}
  ]
}

Kurallar:
- Mümkünse 2-5 tema üret.
- Her öğe tam olarak bir kez yer almalı.
- Tema isimleri kısa, teknik ve Türkçe olmalı.
- Öncelik verilecek lensler (uygunsa): agentic SDLC, on-prem verimlilik/SLM, güvenlik-yönetişim.
- Aynı haber birden fazla temaya yazılmamalı.
- Ek anahtar üretme.

Items:
"""
    user_prompt += "\n".join(payload_lines)

    try:
        data = generate_json(config=config, system=SYSTEM_PROMPT, user=user_prompt)
        themes = data.get("themes", [])
        groups: List[ThemeGroup] = []
        used = set()
        for theme in themes:
            name = str(theme.get("name", "Genel")).strip() or "Genel"
            item_ids = [int(i) for i in theme.get("item_ids", [])]
            used.update(item_ids)
            groups.append(ThemeGroup(name=name, item_ids=item_ids))

        if len(used) != len(items):
            return _fallback(items)

        return groups
    except (ValueError, KeyError, LLMError):
        return _fallback(items)


def _fallback(items: List[SummaryItem]) -> List[ThemeGroup]:
    return [ThemeGroup(name="Genel", item_ids=list(range(len(items))))]
