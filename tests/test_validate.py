from datetime import date

from src.config import ExcludedItem, SummaryItem
from src.validate import deterministic_checks


def _item(url: str) -> SummaryItem:
    return SummaryItem(
        url=url,
        origin_url=url,
        source_name="Source",
        title="Title",
        date=date(2026, 1, 2),
        date_inferred=False,
        summary_tr="Özet.",
        why_it_matters_tr="Neden önemli.",
        tags=["genai"],
        confidence=0.8,
    )


def test_missing_link_detected():
    items = [_item("https://example.com/a")]
    report = "# Haftalık GenAI Raporu (01-01-2026 – 08-01-2026)\n\n## Özet\n- ...\n\n## Haftanın Temaları\n- Tema\n\n## Tema: Tema\n\n### Title\n- Tarih: 02-01-2026\n- Kaynak: Source\n- Gelişme: Özet\n- Neden Önemli: Neden\n\n## Kaynaklar\n"
    issues = deterministic_checks(
        report_text=report,
        items=items,
        input_urls=["https://example.com/a"],
        excluded=[],
    )
    assert any("Missing inline link" in issue for issue in issues)
