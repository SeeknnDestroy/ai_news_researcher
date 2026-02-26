
import re
from pathlib import Path

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

def _find_marker(text: str, marker: str, start: int = 0):
    if not marker:
        return None
    # Simulate the logic in src/newsletter.py
    pattern = re.escape(marker)
    pattern = pattern.replace(r"\ ", r"\s+")
    
    print(f"Searching for marker: '{marker}'")
    print(f"Pattern: '{pattern}'")
    
    match = re.search(pattern, text[start:], flags=re.IGNORECASE)
    if not match:
        print("Match NOT found")
        return None
    print(f"Match found at {start + match.start()}")
    return start + match.start()

def main():
    # Load the text from the debug file that caused issues
    path = Path("artifacts/debug_inputs/05-02-2026_100641/www-deeplearning-ai-the-batch-codex-app-bypasses-cursor-vs-code-on-mac-6ab44027.txt")
    if not path.exists():
        # Fallback to the raw live file if debug not found (similar content)
        path = Path("artifacts/raw_live/codex_app.md")
    
    print(f"Reading from {path}")
    text = path.read_text(encoding="utf-8")
    
    # Extract the body text (skip headers in debug file if present)
    # The debug file has "Title: ...\nURL: ...\n\n" at the top.
    # We need to strip that to match what _trim_newsletter_text sees on the original crawl item text
    # But wait, the debug file IS the crawl item text passed to _write_debug_input.
    # The crawl item text usually starts with the content. 
    # Let's check if the debug file content matches what split_newsletter_items receives.
    # In cli.py: _write_debug_input writes "Title: ...\nURL: ...\n\n{item.text}"
    # So we need to remove the first 3 lines to get item.text.
    
    lines = text.splitlines()
    if lines[0].startswith("Title:"):
        item_text = "\n".join(lines[3:]) # Skip Title, URL, empty line
    else:
        item_text = text

    print(f"Item text length: {len(item_text)}")
    
    trimmed = _trim_newsletter_text(item_text)
    print(f"Trimmed text length: {len(trimmed)}")
    
    # Test the marker that failed
    # [10:08:25] SPLIT_DEBUG | Could not find start marker: 'OpenAI released a macOS applic...'
    # I'll try a few variations that the LLM might have output
    
    markers_to_test = [
        "OpenAI released a macOS application",
        "OpenAI released a macOS application that lets developers",
        "OpenAI released a macOS applic", # In case it was truncated in log but LLM sent full
    ]
    
    for m in markers_to_test:
        _find_marker(trimmed, m)

if __name__ == "__main__":
    main()
