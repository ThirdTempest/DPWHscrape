import os
import sys
import re
import time
from dotenv import load_dotenv
from supabase import create_client
from neo4j import GraphDatabase

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

_supabase = None
_driver = None

def get_clients():
    """Lazily initialize Supabase client and Neo4j driver with friendly error handling."""
    global _supabase, _driver
    if _supabase is None or _driver is None:
        supa_url = os.getenv("SUPABASE_URL")
        supa_key = os.getenv("SUPABASE_SERVICE_KEY")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_pass = os.getenv("NEO4J_PASSWORD")

        if not all([supa_url, supa_key, neo4j_uri, neo4j_pass]):
            print("\n" + "=" * 68)
            print("❌ CONFIGURATION ERROR: Missing Sync Credentials")
            print("=" * 68)
            print("Syncing Supabase projects to Neo4j requires both database connections:")
            print("\n👉 Please check your .env file:")
            if not supa_url or not supa_key:
                print("  • SUPABASE_URL / SUPABASE_SERVICE_KEY are missing (https://supabase.com)")
            if not neo4j_uri or not neo4j_pass:
                print("  • NEO4J_URI / NEO4J_PASSWORD are missing (https://neo4j.com/aura)")
            print("=" * 68 + "\n")
            return None, None

        _supabase = create_client(supa_url, supa_key)
        _driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    return _supabase, _driver

def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def extract_field(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""

def parse_markdown_project(markdown: str, url: str) -> dict | None:
    pid_m = re.search(r'### DPWH Project\s+([A-Za-z0-9]+)', markdown)
    if not pid_m:
        return None

    pid = pid_m.group(1).strip()
    loc = extract_field(r'\*\*Exact Location:\*\*\s*(.*)', markdown)
    off = extract_field(r'\*\*Implementing Office:\*\*\s*(.*)', markdown)
    desc = extract_field(r'\*\*Description:\*\*\s*(.*)', markdown)
    cat = extract_field(r'\*\*Category:\*\*\s*(.*)', markdown)
    con = extract_field(r'\*\*Contractor:\*\*\s*(.*)', markdown)
    bud = extract_field(r'\*\*Budget:\*\*\s*(.*)', markdown)
    stat = extract_field(r'\*\*Status:\*\*\s*(.*)', markdown)
    time_val = extract_field(r'\*\*Timeline:\*\*\s*(.*)', markdown)

    # Parse contractor code if present: 'NAME (12345)'
    con_name, con_code = con, ""
    con_match = re.search(r'^(.*?)\s*\((\d+)\)$', con)
    if con_match:
        con_name = con_match.group(1).strip()
        con_code = con_match.group(2).strip()

    # Parse budget numeric and year: '₱4,935,118 (Year: 2024)'
    bud_num = 0
    bud_num_m = re.search(r'[\u20b1P\$\s]*([0-9,]+)', bud)
    if bud_num_m:
        try:
            bud_num = int(bud_num_m.group(1).replace(",", ""))
        except Exception:
            pass

    year_m = re.search(r'Year:\s*(\d{4})', bud)
    year = int(year_m.group(1)) if year_m else None

    # Parse completion percentage from status if available
    pct_m = re.search(r'(\d+(?:\.\d+)?)%\s*complete', stat, re.IGNORECASE)
    completion_pct = float(pct_m.group(1)) if pct_m else None

    return {
        "project_id": pid,
        "entity_id": f"DPWH Project {pid}",
        "location": loc,
        "office": off,
        "description": desc or f"DPWH Project {pid}",
        "category": cat or "General Infrastructure",
        "contractor_name": con_name,
        "contractor_code": con_code,
        "budget_raw": bud,
        "budget_numeric": bud_num,
        "year": year,
        "status": stat,
        "completion_pct": completion_pct,
        "timeline": time_val,
        "url": url,
    }

def init_neo4j_schema(session):
    """Ensure indexes and constraints exist in Neo4j."""
    statements = [
        "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
        "CREATE INDEX base_entity_id_idx IF NOT EXISTS FOR (n:base) ON (n.entity_id)",
        "CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS FOR (n:base) ON EACH [n.entity_id, n.description, n.location, n.contractor]",
    ]
    for stmt in statements:
        try:
            session.run(stmt)
        except Exception:
            pass

def sync_projects_to_neo4j(reset: bool = False, limit: int = None):
    supabase, driver = get_clients()
    if not supabase or not driver:
        return

    start_time = time.time()

    # 1. Fetch from Supabase with pagination
    print("📥 Fetching scraped DPWH pages from Supabase...")
    rows = []
    page_size = 1000
    offset = 0
    while True:
        query = supabase.table("scraped_pages").select("id, url, markdown").range(offset, offset + page_size - 1)
        if limit and len(rows) + page_size > limit:
            query = query.limit(limit - len(rows))
        res = query.execute()
        batch = res.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size or (limit and len(rows) >= limit):
            break
        offset += page_size
    print(f"📄 Found {len(rows)} pages in Supabase.")

    parsed_projects = []
    skipped_urls = []
    for r in rows:
        parsed = parse_markdown_project(r.get("markdown", ""), r.get("url", ""))
        if parsed:
            parsed["row_id"] = r["id"]
            parsed_projects.append(parsed)
        else:
            skipped_urls.append(r.get("url"))

    print(f"🔍 Successfully parsed {len(parsed_projects)} project detail pages.")

    # 2. Ingest into Neo4j
    with driver.session() as session:
        init_neo4j_schema(session)

        if reset:
            print("🧹 Resetting Neo4j database...")
            session.run("MATCH (n) DETACH DELETE n")

        print("⚡ Ingesting knowledge graph into Neo4j Aura (Batch mode)...")

        batch_cypher = """
        UNWIND $batch AS p

        // 1. Project Node (Tagged with both :Project and :base for LightRAG)
        MERGE (proj:Project:base {entity_id: p.entity_id})
        SET proj.project_id = p.project_id,
            proj.description = p.description,
            proj.location = p.location,
            proj.contractor = p.contractor_name,
            proj.office = p.office,
            proj.budget_raw = p.budget_raw,
            proj.budget_numeric = p.budget_numeric,
            proj.year = p.year,
            proj.category = p.category,
            proj.status = p.status,
            proj.completion_pct = p.completion_pct,
            proj.timeline = p.timeline,
            proj.url = p.url,
            proj.entity_type = "project",
            proj.weight = 1.0

        // 2. Contractor Node & Relationship
        FOREACH (_ IN CASE WHEN p.contractor_name <> "" THEN [1] ELSE [] END |
            MERGE (c:Contractor:base {entity_id: p.contractor_name})
            SET c.name = p.contractor_name,
                c.code = p.contractor_code,
                c.entity_type = "contractor",
                c.description = "Contractor " + p.contractor_name + (CASE WHEN p.contractor_code <> "" THEN " (Code: " + p.contractor_code + ")" ELSE "" END)
            MERGE (proj)-[r:CONTRACTED_TO]->(c)
            SET r.description = p.contractor_name + " is the contractor for " + p.entity_id,
                r.weight = 1.0,
                r.keywords = "contractor, builder, " + p.project_id
        )

        // 3. Location Node & Relationship
        FOREACH (_ IN CASE WHEN p.location <> "" THEN [1] ELSE [] END |
            MERGE (loc:Location:base {entity_id: p.location})
            SET loc.name = p.location,
                loc.entity_type = "location",
                loc.description = "Location: " + p.location
            MERGE (proj)-[r:LOCATED_IN]->(loc)
            SET r.description = p.entity_id + " is located in " + p.location,
                r.weight = 1.0,
                r.keywords = "location, barangay, city, " + p.project_id
        )

        // 4. Implementing Office Node & Relationship
        FOREACH (_ IN CASE WHEN p.office <> "" THEN [1] ELSE [] END |
            MERGE (off:Office:base {entity_id: p.office})
            SET off.name = p.office,
                off.entity_type = "organization",
                off.description = "Implementing Office: " + p.office
            MERGE (proj)-[r:SUPERVISED_BY]->(off)
            SET r.description = p.office + " is the implementing office for " + p.entity_id,
                r.weight = 1.0,
                r.keywords = "office, implementing office, DPWH, " + p.project_id
        )

        // 5. Category Node & Relationship
        FOREACH (_ IN CASE WHEN p.category <> "" THEN [1] ELSE [] END |
            MERGE (cat:Category:base {entity_id: p.category})
            SET cat.name = p.category,
                cat.entity_type = "category",
                cat.description = "Category: " + p.category
            MERGE (proj)-[r:CATEGORIZED_AS]->(cat)
            SET r.description = p.entity_id + " falls under category " + p.category,
                r.weight = 1.0,
                r.keywords = "category, " + p.category
        )
        """

        batch_size = 50
        for i in range(0, len(parsed_projects), batch_size):
            chunk = parsed_projects[i : i + batch_size]
            session.run(batch_cypher, batch=chunk)
            print(f"  • Ingested {min(i + batch_size, len(parsed_projects))}/{len(parsed_projects)} projects...")

        node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        project_count = session.run("MATCH (p:Project) RETURN count(p) as c").single()["c"]
        contractor_count = session.run("MATCH (c:Contractor) RETURN count(c) as c").single()["c"]

    # 3. Mark graph_synced in Supabase
    print("📝 Updating sync flags in Supabase...")
    supabase.table("scraped_pages").update({"graph_synced": True}).neq("url", "").execute()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 SYNC COMPLETE in {elapsed:.2f} seconds!")
    print(f"📊 Neo4j Knowledge Graph Statistics:")
    print(f"   • Total Nodes: {node_count}")
    print(f"   • Total Relationships: {rel_count}")
    print(f"   • Projects Indexed: {project_count}")
    print(f"   • Unique Contractors: {contractor_count}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    sync_projects_to_neo4j(reset=reset_flag)
    close_driver()

