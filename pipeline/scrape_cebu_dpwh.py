import os
import sys
import hashlib
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import httpx
from apify_client import ApifyClient
from supabase import create_client

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([APIFY_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables in .env")

apify = ApifyClient(APIFY_TOKEN)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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

def extract_location_from_description(desc: str, default_deo: str) -> str:
    """Extract barangay, city, or municipal name from the project description."""
    m = re.search(
        r'(?:IN|AT|ALONG|BARANGAY|BRGY\.?)\s+([A-Z0-9\s,\.\(\)\-]+?,\s*(?:CEBU(?:\s+CITY)?|MANDAUE(?:\s+CITY)?|LAPU[- ]LAPU(?:\s+CITY)?|TALISAY(?:\s+CITY)?|TOLEDO(?:\s+CITY)?|BOGO(?:\s+CITY)?|CARCAR(?:\s+CITY)?|NAGA(?:\s+CITY)?|DANAO(?:\s+CITY)?))',
        desc,
        re.IGNORECASE,
    )
    if m:
        loc = m.group(1).strip()
        loc = re.sub(r'\s+', ' ', loc)
        return f"{loc}, CEBU" if not loc.upper().endswith("CEBU") else loc

    for city in [
        "CEBU CITY", "MANDAUE CITY", "LAPU-LAPU CITY", "TALISAY CITY",
        "TOLEDO CITY", "BOGO CITY", "DANAO CITY", "CARCAR CITY", "NAGA CITY"
    ]:
        if city in desc.upper():
            return f"{city}, CEBU"

    jurisdiction = CEBU_DEO_MAP.get(default_deo, "Cebu")
    return f"{jurisdiction}, CEBU"

def fetch_all_cebu_dpwh_projects(max_projects: int = None) -> list[dict]:
    """Fetch all DPWH Cebu projects from 2022 to present via paginated Meilisearch queries."""
    headers = {
        "Authorization": "Bearer 307c9f43a066a443cc37d62b45fa47fde2b39f765139dd964ea151daed65f55c",
        "Content-Type": "application/json"
    }
    
    all_hits = []
    offset = 0
    page_limit = 1000

    print("📡 Fetching DPWH Cebu projects (2022-present) from BetterGov search index...")

    while True:
        payload = {
            "q": "Cebu",
            "filter": "infraYear IN ['2022', '2023', '2024', '2025', '2026']",
            "limit": page_limit,
            "offset": offset
        }

        r = httpx.post(
            "https://search2.bettergov.ph/indexes/dpwh/search",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", [])
        if not hits:
            break

        all_hits.extend(hits)
        total_est = data.get("totalHits") or data.get("estimatedTotalHits") or len(all_hits)
        print(f"  • Retrieved batch: {len(hits)} projects (Total accumulated: {len(all_hits)} / ~{total_est})")

        if max_projects and len(all_hits) >= max_projects:
            all_hits = all_hits[:max_projects]
            break

        if len(hits) < page_limit:
            break

        offset += page_limit

    print(f"✅ Total fetched: {len(all_hits)} Cebu infrastructure projects.")
    return all_hits

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape ALL Cebu DPWH Projects (2022-present) to Apify, Supabase, and Neo4j")
    parser.add_argument("--all", "-a", action="store_true", default=True, help="Scrape all available projects (default: True)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Optional max project limit (e.g. 1000)")
    parser.add_argument("--no-sync", action="store_true", help="Skip automatic sync to Neo4j")
    args = parser.parse_args()

    # 1. Fetch from BetterGov Meilisearch API
    projects = fetch_all_cebu_dpwh_projects(max_projects=args.limit)
    if not projects:
        print("❌ No projects fetched.")
        return

    # 2. Push to Apify Dataset
    dataset_name = f"cebu-dpwh-2022-present-{datetime.now().strftime('%Y%m%d')}"
    dataset = apify.datasets().get_or_create(name=dataset_name)
    dataset_id = dataset.id
    print(f"\n📦 Connected Apify Dataset: {dataset_name} (ID: {dataset_id})")
    print(f"⬆️ Uploading {len(projects)} projects to Apify...")

    # Upload in chunks of 1000
    for i in range(0, len(projects), 1000):
        chunk = projects[i:i + 1000]
        apify.dataset(dataset_id).push_items(chunk)
        print(f"  • Pushed {min(i + 1000, len(projects))}/{len(projects)} items to Apify Dataset...")
    print("✅ Apify Dataset upload complete.")

    # 3. Format into standardized DPWH Markdown and prepare Supabase rows
    print(f"\n💾 Upserting to Supabase 'scraped_pages' table...")
    supabase_rows = []
    for p in projects:
        cid = p.get("contractId")
        if not cid:
            continue

        desc = p.get("description", "DPWH Infrastructure Project")
        deo = p.get("location", {}).get("province", "Cebu DEO")
        category = p.get("category") or p.get("componentCategories") or "Infrastructure"
        contractor = p.get("contractor") or "Unspecified Contractor"
        budget = p.get("budget", 0)
        status = p.get("status", "Completed")
        progress = p.get("progress", 100.0)
        year = p.get("infraYear", "2024")
        start = p.get("startDate") or "N/A"
        end = p.get("completionDate") or "N/A"
        url = f"https://transparency.bettergov.ph/dpwh/projects/{cid}"

        loc_str = extract_location_from_description(desc, deo)
        jurisdiction = CEBU_DEO_MAP.get(deo, deo)
        office_str = f"{deo} [Jurisdiction: {jurisdiction}]"

        markdown_content = (
            f"### DPWH Project {cid}\n"
            f"- **Category:** {category}\n"
            f"- **Exact Location:** {loc_str}\n"
            f"- **Implementing Office:** {office_str}\n"
            f"- **Contractor:** {contractor}\n"
            f"- **Budget:** ₱{budget:,.2f} (Year: {year})\n"
            f"- **Status:** {status} ({progress:.1f}% complete)\n"
            f"- **Timeline:** {start} to {end}\n"
            f"- **Description:** {desc}\n"
        )
        content_hash = hashlib.sha256(markdown_content.encode("utf-8")).hexdigest()

        row_data = {
            "url": url,
            "title": f"DPWH Project {cid} - {category}",
            "markdown": markdown_content,
            "metadata": {
                "contractId": cid,
                "infraYear": year,
                "budget": budget,
                "status": status,
                "progress": progress,
                "deo": deo,
                "contractor": contractor,
                "source": "bettergov_dpwh"
            },
            "content_hash": content_hash,
            "graph_synced": False
        }
        supabase_rows.append(row_data)

    # 4. Batch upsert to Supabase in chunks of 100
    batch_size = 100
    upserted_count = 0
    for i in range(0, len(supabase_rows), batch_size):
        chunk = supabase_rows[i:i + batch_size]
        supabase.table("scraped_pages").upsert(chunk, on_conflict="url").execute()
        upserted_count += len(chunk)
        print(f"  • Upserted {upserted_count}/{len(supabase_rows)} projects to Supabase...")

    print(f"\n🎉 Successfully saved {upserted_count} Cebu DPWH projects (2022-present) into Supabase!")

    # 5. Automatically Sync to Neo4j Aura
    if not args.no_sync:
        print("\n🔄 Ingesting entire dataset into Neo4j Aura knowledge graph...")
        try:
            from pipeline.sync_to_neo4j import sync_projects_to_neo4j
        except ImportError:
            from sync_to_neo4j import sync_projects_to_neo4j
        sync_projects_to_neo4j(reset=False)

if __name__ == "__main__":
    main()
