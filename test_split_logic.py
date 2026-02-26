
import sys
from pathlib import Path
print("Script started")

try:
    from src.newsletter import _trim_newsletter_text, _extract_items
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_codex_app():
    path = Path("artifacts/raw_live/codex_app.md")
    print(f"Checking path: {path.absolute()}")
    if not path.exists():
        print(f"File not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    print(f"Read {len(text)} chars")
    
    trimmed = _trim_newsletter_text(text)
    print(f"Trimmed to {len(trimmed)} chars")
    
    # Simulate splitting (using heuristic fallback)
    items = _extract_items(trimmed, max_items=10)
    
    print(f"Found {len(items)} items:")
    for i, item in enumerate(items, 1):
        print(f"\n[{i}] TITLE: {item['title']}")
        print(f"    URL: {item.get('url', 'None')}")
        print(f"    TEXT START: {item['text'][:100].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    test_codex_app()
