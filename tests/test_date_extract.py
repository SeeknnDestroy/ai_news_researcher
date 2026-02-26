from datetime import date

from src.date_extract import extract_date, is_within_window


def test_extract_from_metadata():
    metadata = {"og:published_time": "2026-01-05T10:00:00Z"}
    result = extract_date(metadata, "", "https://example.com")
    assert result.value == date(2026, 1, 5)
    assert result.inferred is False


def test_extract_from_text():
    text = "This article was published on 2026-01-06 and updated later."
    result = extract_date({}, text, "https://example.com")
    assert result.value == date(2026, 1, 6)
    assert result.inferred is True


def test_extract_from_url():
    url = "https://example.com/2026/01/07/article"
    result = extract_date({}, "", url)
    assert result.value == date(2026, 1, 7)


def test_window_plus_minus_one():
    assert is_within_window(date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 2))
    assert is_within_window(date(2026, 1, 3), date(2026, 1, 2), date(2026, 1, 2))
    assert not is_within_window(date(2026, 1, 4), date(2026, 1, 2), date(2026, 1, 2))
