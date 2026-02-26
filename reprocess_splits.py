
import sys
import glob
from pathlib import Path
from src.config import CrawlItem
from src.newsletter import split_newsletter_items
from src.cli import _write_split_items
from src.llm import XAIConfig
from src.utils import slugify_url

def main():
    # Use the specific raw directory we found
    raw_dir = Path("artifacts/raw/05-02-2026_100641")
    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        return

    # Mock config - we don't actually need valid API keys if we fall back to heuristic
    # BUT, the splitting logic tries LLM first. 
    # If I want to rely on the heuristic fallback (which worked in my small test), 
    # I can let LLM fail or mock it.
    # However, I should try to use the real config if available so LLM might work?
    # No, I'll rely on the fallback logic I added/debugged which uses regex/markers 
    # or the heuristic extraction.
    # Actually, the user's issue was "marker not found".
    # I improved the marker finding logic.
    # So if I run this, it will try LLM (if configured) or fail to LLM then use fallback.
    
    # Let's assume environment variables for keys are loaded if I use src.cli logic?
    # I'll just init config.
    config = XAIConfig(model="mock") # Model doesn't matter if we don't call it or if we expect it to fail/mock

    run_id = "reprocess_fixed"
    
    print(f"Processing files in {raw_dir}...")
    
    files = list(raw_dir.glob("*.txt"))
    for file_path in files:
        print(f"\nProcessing {file_path.name}...")
        text = file_path.read_text(encoding="utf-8")
        
        # We need to reconstruct the URL. 
        # The filename is slugified. We can try to guess or just use the filename as URL for splitting purposes.
        # Actually, the file content in raw/ doesn't have metadata headers in the file itself (based on _write_raw_text implementation).
        # _write_raw_text just writes item.text.
        
        # However, for the splitting to work, we need a URL that looks like a newsletter URL 
        # to pass _is_newsletter check if it's strict.
        # But I added logging to see if it skips.
        
        # Let's try to extract URL from the file content if possible, or use a dummy "newsletter" url.
        url = "https://www.deeplearning.ai/the-batch/" + file_path.stem
        
        item = CrawlItem(
            url=url,
            text=text,
            metadata={},
            title="Reprocessed Item",
            origin_url=url
        )
        
        # Run splitting
        # Note: This will try LLM. If no API key, it raises error or returns empty?
        # generate_json raises LLMError.
        # _llm_split_markers catches LLMError and returns [].
        # So it will fall back to heuristic immediately.
        derived_items = split_newsletter_items(config, item, max_items=10)
        
        if len(derived_items) > 1:
            print(f"  -> Split into {len(derived_items)} items")
            paths = _write_split_items("reports/dummy.md", run_id, url, derived_items)
            print(f"  -> Saved to {Path(paths[0]).parent}")
        else:
            print("  -> No split occurred (or single item returned)")

if __name__ == "__main__":
    main()
