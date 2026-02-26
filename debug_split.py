
import sys
import re
from pathlib import Path

# Copying the logic from src/newsletter.py to debug it here
def _trim_newsletter_text(text: str) -> str:
    if not text:
        return ""
    stop_markers = [
        "want to know more",
        "a special offer",
        "subscribe",
        "try pro membership",
        "enroll now",
    ]
    lowered = text.lower()
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

def _first_link(text: str) -> str:
    match = re.search(r"\((https?://[^)]+)\)", text)
    if match:
        return match.group(1)
    match = re.search(r"(https?://\S+)", text)
    return match.group(1) if match else ""

def _extract_items(text: str, max_items: int) -> list:
    if not text:
        return []

    # Regex for **Heading**
    # We want to see if this matches
    pattern = r"\*\*(.+?)\*\*"
    headings = list(re.finditer(pattern, text))
    print(f"DEBUG: Found {len(headings)} headings with pattern {pattern!r}")
    
    items = []
    if headings:
        for i, match in enumerate(headings):
            title = match.group(1).strip()
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            url = _first_link(body)
            items.append({"title": title, "url": url, "text": body})
            if len(items) >= max_items:
                break
        return items
    
    return []

def main():
    path = Path("artifacts/raw_live/codex_app.md")
    print(f"Reading {path}")
    text = path.read_text(encoding="utf-8")
    
    trimmed = _trim_newsletter_text(text)
    print(f"Trimmed size: {len(trimmed)}")
    
    # Check if headings exist in trimmed text
    print("Sample trimmed text (start):")
    print(trimmed[:200])
    
    items = _extract_items(trimmed, 10)
    print(f"Found {len(items)} items")
    for item in items:
        print(f"- {item['title']}")

if __name__ == "__main__":
    main()
