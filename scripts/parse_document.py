# scripts/parse_document.py
import os
import sys
import json
import re
from bs4 import BeautifulSoup

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def parse_html_to_json(filepath, lang):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract all paragraphs and headers
    elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'div'])

    structured_data = []
    current_section = "Introduction"

    # Regex to catch "Article X" or "Artikel X"
    article_pattern = re.compile(r'^(Article|Artikel)\s+\d+', re.IGNORECASE)

    for el in elements:
        text = el.get_text(strip=True)
        if not text:
            continue

        # If the text is a header for a new Article, update the state
        if article_pattern.match(text):
            current_section = text
            continue # Skip adding the title alone as a chunk

        # Only keep meaningful paragraphs (skip very short boilerplate)
        if len(text) > 40:
            structured_data.append({
                "document": "EU AI Act",
                "language": lang,
                "section": current_section,
                "content": text,
                "source": f"EUR-Lex ({lang.upper()})"
            })

    return structured_data

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for lang in ["en", "de"]:
        raw_path = os.path.join(RAW_DATA_DIR, f"ai_act_{lang}.html")
        processed_path = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_parsed.json")

        if os.path.exists(raw_path):
            print(f"Parsing {lang.upper()} document...")
            parsed_data = parse_html_to_json(raw_path, lang)

            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {len(parsed_data)} chunks to {processed_path}")
        else:
            print(f"❌ Raw file not found: {raw_path}")

if __name__ == "__main__":
    main()
