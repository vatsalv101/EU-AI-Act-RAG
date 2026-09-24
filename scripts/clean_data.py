# scripts/clean_data.py
import os
import sys
import json
import re

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def clean_text(text):
    """Normalize whitespace and remove newline artifacts."""
    # Replace multiple spaces or newlines with a single space
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_chunk(text):
    """Rule-based filtering for legalese noise."""
    # 1. Filter chunks that are too short to contain semantic value for RAG
    if len(text.split()) < 8:
        return False

    # 2. Filter standard EU Official Journal footnotes (e.g., "(1) OJ L 285...")
    if re.match(r'^\(\d+\)\s*OJ\s+L', text, re.IGNORECASE):
        return False

    # 3. Filter purely numeric or punctuation chunks
    if re.match(r'^[\d\W]+$', text):
        return False

    return True

def clean_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)
    cleaned_data = []

    for item in data:
        cleaned_content = clean_text(item["content"])

        if is_valid_chunk(cleaned_content):
            item["content"] = cleaned_content
            cleaned_data.append(item)

    return original_count, cleaned_data

def main():
    for lang in ["en", "de"]:
        input_path = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_parsed.json")
        output_path = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_clean.json")

        if os.path.exists(input_path):
            print(f"Cleaning {lang.upper()} dataset...")
            original_count, clean_data = clean_dataset(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, ensure_ascii=False, indent=2)

            removed = original_count - len(clean_data)
            print(f"✅ Cleaned {lang.upper()}: Removed {removed} noisy chunks. Kept {len(clean_data)} chunks.")
            print(f"   Saved to {output_path}")

if __name__ == "__main__":
    main()
