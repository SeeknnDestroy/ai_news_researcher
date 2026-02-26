from __future__ import annotations

from typing import List

from .config import ExcludedItem, SummaryItem, ThemeGroup
from .llm import XAIConfig, LLMError, generate_json
from .utils import format_date


SUMMARY_SYSTEM_PROMPT = """
You are a senior Turkish technical editor writing an executive+engineering weekly GenAI brief.
Style must be dense, clinical, concise, and non-promotional.
Prefer concrete facts and strategic implications over generic wording.
Return ONLY valid JSON.
""".strip()


def synthesize_report(
    config: XAIConfig,
    items: List[SummaryItem],
    themes: List[ThemeGroup],
    excluded: List[ExcludedItem],
) -> str:
    summary_lines = _generate_summary(config, items)
    theme_names = [theme.name for theme in themes]

    lines: List[str] = []
    lines.append("# Haftalık GenAI Raporu")
    lines.append("")
    lines.append("## Özet")
    for bullet in summary_lines:
        lines.append(f"- {bullet}")

    lines.append("")
    lines.append("## Haftanın Temaları")
    for name in theme_names:
        lines.append(f"- {name}")

    lines.append("")
    for theme in themes:
        lines.append(f"## Tema: {theme.name}")
        lines.append("")
        for idx in theme.item_ids:
            item = items[idx]
            inferred = " (tahmini)" if item.date_inferred else ""
            lines.append(f"### <u>**{item.title}**</u>")
            lines.append(f"* **Tarih:** {format_date(item.date)}{inferred}")
            lines.append(f"* **Kaynak:** [{item.source_name}]({item.url})")
            lines.append(f"* **Gelişme:** {item.summary_tr}")
            lines.append(f"* **Neden Önemli:** {item.why_it_matters_tr}")
            lines.append("")

    lines.append("## Kaynaklar")
    for item in items:
        lines.append(f"- [{item.source_name}]({item.url})")

    if excluded:
        lines.append("")
        lines.append("## Kullanılamayan Kaynaklar")
        for entry in excluded:
            lines.append(f"- {entry.url} — {entry.reason}")

    return "\n".join(lines).strip() + "\n"


def _generate_summary(config: XAIConfig, items: List[SummaryItem]) -> List[str]:
    if not items:
        return ["Bu hafta rapora dahil edilebilir kaynak bulunamadı."]

    bullet_count = _summary_bullet_count(len(items))
    payload = "\n".join(
        f"- {item.title}: {item.summary_tr}" for item in items[:10]
    )
    user_prompt = f"""
Return JSON: {{"bullets": ["..."]}}

Write {bullet_count} Turkish bullets to summarize the week.
Rules:
- Max 22 words each.
- Technical and strategic (no hype).
- Include impact signals (benchmark/cost/speed/risk) when available.
- Do not mention URLs.
- Do not repeat the same claim across bullets.

Items:
{payload}
"""

    try:
        data = generate_json(config=config, system=SUMMARY_SYSTEM_PROMPT, user=user_prompt)
        bullets = [str(b).strip() for b in data.get("bullets", []) if str(b).strip()]
        if len(bullets) >= 1:
            return bullets[:bullet_count]
    except LLMError:
        pass

    # Fallback: use titles
    return [item.title for item in items[:bullet_count]]


def _summary_bullet_count(n_items: int) -> int:
    if n_items <= 0:
        return 1
    if n_items <= 3:
        return n_items
    if n_items <= 5:
        return 3
    if n_items <= 8:
        return 4
    if n_items <= 12:
        return 5
    return 6
