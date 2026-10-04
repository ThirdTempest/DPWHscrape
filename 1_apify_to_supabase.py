import os
import sys
import hashlib
import re
from dotenv import load_dotenv
from apify_client import ApifyClient
from supabase import create_client

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()  # <-- Reads your .env file!

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([APIFY_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables in .env (APIFY_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY)")

apify = ApifyClient(APIFY_TOKEN)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 1. Get the dataset from your last Website Content Crawler run
runs = apify.actor("apify/playwright-scraper").runs().list(desc=True, limit=1)
if not runs.items:
    print("❌ No Apify runs found! Make sure you clicked 'Save & start' in Apify first.")
    exit()

dataset_id = runs.items[0].default_dataset_id
print(f"📥 Pulling scraped pages from Apify Dataset: {dataset_id}...")

# 2. Loop through each scraped page and save to Supabase
count = 0
for item in apify.dataset(dataset_id).iterate_items():
    url = item.get("url")
    markdown = item.get("markdown")
    metadata = item.get("metadata") or {}
    title = metadata.get("title", "Untitled Page")

    if not url or not markdown:
        continue

    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    row_data = {
        "url": url,
        "title": title,
        "markdown": markdown,
        "metadata": metadata,
        "content_hash": content_hash,
        "graph_synced": False
    }

    supabase.table("scraped_pages").upsert(row_data, on_conflict="url").execute()
    count += 1
    print(f"✅ [{count}] Saved to Supabase: {title}")

print(f"\n🎉 Done! {count} pages are now in your Supabase 'scraped_pages' table.")

CEBU_DEO_MAP = {
    "Cebu City DEO": "Cebu City (North & South Districts)",
    "Cebu 1st DEO": "Talisay City, Minglanilla, Naga City, San Fernando, Carcar City, Sibonga",
    "Cebu 2nd DEO": "Argao, Dalaguete, Alcoy, Boljoon, Oslob, Santander, Samboan",
    "Cebu 3rd DEO": "Toledo City, Balamban, Asturias, Tuburan, Pinamungajan, Aloguinsan, Barili",
    "Cebu 4th DEO": "Bogo City, San Remigio, Medellin, Daanbantayan, Santa Fe, Bantayan, Madridejos, Tabogon, Tabuelan",
    "Cebu 5th DEO": "Danao City, Compostela, Liloan, Carmen, Catmon, Sogod, Borbon, Camotes Islands",
    "Cebu 6th DEO": "Lapu-Lapu City, Mandaue City, Consolacion, Cordova",
    "Cebu 7th DEO": "Dumanjug, Ronda, Alcantara, Moalboal, Badian, Alegria, Malabuyoc, Ginatilan",
}

# Also map the 2-letter Contract ID codes (e.g., 20HN0111 -> HN = Cebu 6th DEO)
CONTRACT_PREFIX_MAP = {
    "HH": "Cebu City DEO (Cebu City)",
    "HE": "Cebu 1st DEO (Talisay, Minglanilla, Naga, San Fernando, Carcar, Sibonga)",
    "HG": "Cebu 2nd DEO (Argao, Dalaguete, Alcoy, Boljoon, Oslob, Santander, Samboan)",
    "HF": "Cebu 3rd DEO (Toledo, Balamban, Asturias, Tuburan, Pinamungajan, Aloguinsan, Barili)",
    "HD": "Cebu 4th DEO (Bogo, San Remigio, Medellin, Daanbantayan, Santa Fe, Bantayan, Madridejos, Tabogon, Tabuelan)",
    "HI": "Cebu 5th DEO (Danao, Compostela, Liloan, Carmen, Catmon, Sogod, Borbon, Camotes)",
    "HN": "Cebu 6th DEO (Lapu-Lapu City, Mandaue City, Consolacion, Cordova)",
    "HJ": "Cebu 7th DEO (Dumanjug, Ronda, Alcantara, Moalboal, Badian, Alegria, Malabuyoc, Ginatilan)",
}

def enrich_cebu_locations(markdown_text: str) -> str:
    """Replaces generic Cebu DEO labels with explicit Jurisdiction Location tags for GraphRAG."""
    enriched = markdown_text
    for deo, jurisdiction in CEBU_DEO_MAP.items():
        if deo in enriched:
            enriched = enriched.replace(
                deo, f"{deo} [Jurisdiction: {jurisdiction}]"
            )
    return enriched