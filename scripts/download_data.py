# scripts/download_data.py
import os
import sys
import requests

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The CELEX number for the EU AI Act is 32024R1689
URLS = {
    "en": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
    "de": "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32024R1689"
}

# Official Publications Office Cellar direct endpoints (fallback if EUR-Lex WAF challenges automated requests)
FALLBACK_URLS = {
    "en": "http://publications.europa.eu/resource/cellar/dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03/DOC_1",
    "de": "http://publications.europa.eu/resource/cellar/dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0004.03/DOC_1"
}

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def download_html():
    # Ensure the directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    for lang, url in URLS.items():
        print(f"Downloading {lang.upper()} version...")
        response = requests.get(url, headers=HEADERS)

        # Check if EUR-Lex returned 200 or if CloudFront WAF challenged it (e.g. 202 status)
        content = None
        if response.status_code == 200 and len(response.text) > 50000:
            content = response.text
        else:
            # Fall back to the EU Publications Office Cellar repository
            fallback_url = FALLBACK_URLS.get(lang)
            if fallback_url:
                fb_resp = requests.get(fallback_url, headers=HEADERS)
                if fb_resp.status_code == 200 and len(fb_resp.text) > 50000:
                    content = fb_resp.text

        if content:
            filepath = os.path.join(RAW_DATA_DIR, f"ai_act_{lang}.html")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved to {filepath}")
        else:
            print(f"❌ Failed to download {lang}. Status code: {response.status_code}")

if __name__ == "__main__":
    download_html()
