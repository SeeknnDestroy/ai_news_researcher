from __future__ import annotations

from .config import CrawlItem, SummaryItem
from .llm import XAIConfig, generate_json
from .utils import clamp_text, source_name_from_url


SYSTEM_PROMPT = """
You are a senior AI research analyst preparing a weekly GenAI technical report.

Mission:
- Produce high-signal, decision-support summaries for executives, tech leads, and AI engineers.
- Prioritize engineering impact over storytelling.

Strict style rules:
- Output language: Turkish.
- Tone: clinical, technical, concise, authoritative.
- No marketing hype. Never use: "game changer", "new era", "revolutionary", "unleashed", "mind-blowing".
- Prefer concrete facts (benchmarks, costs, latency, parameter size, architecture terms) when present.

Grounding rules:
- Use only information present in the provided article text.
- Do not invent numbers, benchmark names, claims, organizations, timelines, or URLs.
- If a requested detail is not clearly supported, omit it or mark uncertainty briefly.
- Do not force sector-specific framing.

Priority lenses (apply only when relevant to the source):
1) Agentic Coding & SDLC transformation.
2) On-premise feasibility, SLM efficiency, quantization, open weights, sovereignty.
3) Security, governance, prompt-injection, shadow AI, enterprise controls.

Return ONLY valid JSON.
""".strip()


def summarize_article(config: XAIConfig, item: CrawlItem, audience: str = "mixed") -> SummaryItem:
    source_name = _source_name(item)
    title = item.title or "(Başlık yok)"
    article_text = clamp_text(item.text, 8000)

    user_prompt = f"""
Görev: Aşağıdaki haberi haftalık GenAI teknik raporu için özetle.

JSON şeması (yalnızca bu anahtarlar):
{{
  "title": "string",
  "source_name": "string",
  "summary_tr": "string",
  "why_it_matters_tr": "string",
  "tags": ["string"],
  "confidence": 0.0
}}

Alan kuralları:
- title: Türkçe, kısa ve etkili. Kaynak başlığı çevrilebilir.
- source_name: kaynak adı.
- summary_tr (Gelişme): 2-3 cümle, yaklaşık 45-90 kelime, teknik ve yoğun; varsa sayısal metrikleri dahil et.
- why_it_matters_tr (Neden Önemli): 1-2 cümle. Stratejik etkisini açıkla.
  - Önce genel teknik/ürün/organizasyon etkisini anlat.
  - Sadece gerçekten ilgiliyse SDLC/verimlilik, on-prem, güvenlik/yönetişim bağlantısı kur.
  - "Banka", "bankacılık", "finans" gibi sektör referanslarını yalnızca kaynak içeriği bunu açıkça destekliyorsa kullan.
  - Kaynakta kanıt yoksa sektör bağı kurma.
- tags: 2-5 adet, küçük harfli kısa etiket.
- confidence: 0-1 arası; metnin açıklık ve kanıt gücüne göre.

Güvenilirlik kuralları:
- Metinde geçmeyen metrik/benchmark/cost bilgisi uydurma.
- Belirsiz alanları kesinmiş gibi yazma.
- Markdown, madde imi, ek alan, açıklama metni ekleme.

Bağlam:
- Audience: {audience} (exec + tech lead + AI engineer karışık)
- Source URL: {item.url}
- Source Name: {source_name}
- Source Title: {title}

Article text:
"""
    user_prompt = user_prompt + article_text

    data = generate_json(config=config, system=SYSTEM_PROMPT, user=user_prompt)

    return SummaryItem(
        url=item.url,
        origin_url=item.origin_url or item.url,
        source_name=str(data.get("source_name") or source_name),
        title=str(data.get("title") or title),
        date=None,  # will be set later
        date_inferred=False,
        summary_tr=str(data.get("summary_tr", "")).strip(),
        why_it_matters_tr=str(data.get("why_it_matters_tr", "")).strip(),
        tags=_safe_tags(data.get("tags")),
        confidence=_clamp_confidence(_safe_float(data.get("confidence", 0.0))),
    )


def _source_name(item: CrawlItem) -> str:
    if item.metadata:
        for key in ("site_name", "og:site_name", "source"):
            if item.metadata.get(key):
                return str(item.metadata.get(key))
    return source_name_from_url(item.url)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for tag in value:
        text = str(tag).strip().lower()
        if not text:
            continue
        cleaned.append(text[:30])
    return cleaned[:5]


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
