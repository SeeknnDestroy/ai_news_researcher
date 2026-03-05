from __future__ import annotations

from typing import List

from .config import SummaryItem, ThemeGroup
from .llm import XAIConfig, LLMError, generate_json
from .templates.prompts import THEMES_SYSTEM_PROMPT, themes_user_prompt


def group_themes(config: XAIConfig, items: List[SummaryItem]) -> List[ThemeGroup]:
    if not items:
        return []

    payload_lines = []
    for idx, item in enumerate(items):
        payload_lines.append(
            f"{idx}. {item.title} | tags: {', '.join(item.tags)} | summary: {item.summary_tr}"
        )

    user_prompt = themes_user_prompt("\n".join(payload_lines))

    try:
        data = generate_json(config=config, system=THEMES_SYSTEM_PROMPT, user=user_prompt)
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
