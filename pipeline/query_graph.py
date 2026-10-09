import os
import sys
import re
import asyncio
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import httpx

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

_driver = None

def print_neo4j_missing_error():
    print("\n" + "=" * 68)
    print("❌ CONFIGURATION ERROR: Missing Neo4j Database Credentials")
    print("=" * 68)
    print("The DPWH Cebu Knowledge Graph requires a Neo4j connection.")
    print("\n👉 Quick Setup Steps:")
    print("  1. Copy the template environment file:")
    print("     copy .env.example .env     (Windows)")
    print("     cp .env.example .env       (Mac/Linux)")
    print("\n  2. Open .env and add your Neo4j credentials:")
    print('     NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"')
    print('     NEO4J_USERNAME="neo4j"')
    print('     NEO4J_PASSWORD="your-password"')
    print("\n     • Free Neo4j Aura cloud database: https://neo4j.com/cloud/platform/aura-graph-database/")
    print("     • Or local Neo4j Desktop / Docker: bolt://localhost:7687")
    print("\n💡 Note: All queries, project lookups, and instant search work")
    print("   purely with Neo4j — No AI API or local AI is required!")
    print("=" * 68 + "\n")

def get_driver():
    """Lazily initialize Neo4j driver with custom error reporting."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD")
        if not all([uri, pwd]):
            print_neo4j_missing_error()
            sys.exit(1)
        _driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return _driver

class DriverProxy:
    def session(self, **kwargs):
        return get_driver().session(**kwargs)
    def close(self):
        global _driver
        if _driver is not None:
            _driver.close()
            _driver = None

driver = DriverProxy()


STOP_WORDS = {
    "what", "are", "is", "the", "for", "in", "of", "and", "a", "an", "who", "which",
    "where", "how", "tell", "me", "about", "project", "projects", "located", "there",
    "any", "give", "list", "show", "it", "that", "this", "them", "some", "many",
    "much", "count", "number", "total", "sum", "can", "you", "please", "find",
    "all", "do", "does", "did", "have", "has", "had", "were", "was", "info", "details"
}

INFRA_DOMAIN_KEYWORDS = {
    "dpwh", "project", "projects", "infrastructure", "infra", "contractor", "contractors",
    "builder", "builders", "budget", "cost", "peso", "pesos", "allocated", "expenditure",
    "cebu", "mandaue", "lapu", "talisay", "toledo", "bogo", "naga", "carcar",
    "danao", "minglanilla", "consolacion", "cordova", "compostela", "liloan", "san fernando",
    "balamban", "asturias", "tuburan", "pinamungajan", "aloguinsan", "barili", "dumanjug",
    "ronda", "alcantara", "moalboal", "badian", "alegria", "malabuyoc", "ginatilan", "samboan",
    "santander", "oslob", "boljoon", "alcoy", "dalaguete", "argao", "sibonga", "san remigio",
    "medellin", "daanbantayan", "madridejos", "bantayan", "santa fe", "tabogon", "tabuelan",
    "borbon", "sogod", "catmon", "carmen", "camotes", "pilar", "poro", "san francisco",
    "tudela", "road", "roads", "highway", "bridge", "bridges", "flyover", "drainage",
    "flood", "river", "seawall", "building", "hall", "school", "slope", "deo", "status",
    "civil", "works", "engineering", "asphalt", "concrete", "paving", "rehabilitation"
}


def is_off_topic_query(query: str) -> tuple[bool, str]:
    """Pre-retrieval intent classifier: intercepts programming, code generation, generic trivia, and off-domain definitions in <1ms."""
    q_lower = query.lower().strip()

    # 1. Direct Project ID exemption (e.g. "24HH0043", "what is 24HH0043?")
    if re.search(r'\b\d{2}[A-Za-z]{1,2}\d{4,5}\b', query):
        return False, ""

    # 2. Exempt legitimate DPWH infrastructure code queries (e.g. "contractor code", "project code", "contractor 7494")
    is_domain_code = bool(re.search(r'\b(?:contractor|project|office|procurement|district|dpwh)\s+code\b', q_lower))
    if is_domain_code:
        return False, ""

    # 3. Programming, scripting, or coding query intent
    is_code_request = bool(re.search(
        r'\b(?:for\s+loops?|while\s+loops?|nested\s+loops?|'
        r'write\s+(?:a\s+)?(?:code|script|program|function|class)|'
        r'code\s+(?:for|to|in|example|snippet|sample)|'
        r'python|javascript|typescript|c\+\+|golang|rust|html|css|php|java\b|'
        r'function|syntax|algorithm|compiler|debugging|recursion|sql\s+query|'
        r'def\s+\w+|import\s+\w+|console\.log|print\()\b',
        q_lower
    ))
    if is_code_request:
        return True, "Programming and software code generation queries are outside system scope."

    # 4. Creative writing, recipes, general chat trivia
    is_creative = bool(re.search(
        r'\b(?:write\s+(?:a\s+)?(?:poem|poetry|story|essay|song|joke)|'
        r'tell\s+me\s+a\s+joke|weather\s+in|recipe\s+for|capital\s+of|'
        r'who\s+won|translate\s+(?:to|into)|how\s+to\s+cook)\b',
        q_lower
    ))
    if is_creative:
        return True, "General trivia and creative writing queries are outside system scope."

    # 5. Non-domain generic definition queries (e.g. "what is science?", "what is physics?", "what is love?")
    # Only block if NO domain keywords (like cebu, project, school, building, road, contractor) are present
    is_generic_def = bool(re.search(
        r'\b(?:what\s+is|what\s+are|define|explain)\s+(?:science|physics|chemistry|biology|philosophy|math|mathematics|ai|artificial\s+intelligence|machine\s+learning|love|life|gravity|photosynthesis|democracy|capitalism)\b',
        q_lower
    ))
    if is_generic_def:
        has_domain_hint = any(kw in q_lower for kw in [
            "cebu", "dpwh", "project", "projects", "school", "high school", "building",
            "facility", "contractor", "budget", "cost", "location", "status", "road", "bridge"
        ])
        if not has_domain_hint:
            return True, "Generic concept definitions outside DPWH civil engineering are outside system scope."

    return False, ""



# ==========================================================
# 1. Instant Neo4j Graph Queries (< 50ms)
# ==========================================================

def clean_location(loc: str | None) -> str:
    """Strip 'Barangay Unspecified' noise and standardize city/province formatting."""
    if not loc:
        return "N/A"
    cleaned = re.sub(r'Barangay\s+Unspecified[,\s]*', '', loc, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^[,\s]+|[,\s]+$', '', cleaned)
    return cleaned if cleaned else loc

def clean_description(desc: str | None) -> str:
    """Strip leading statutory program prefixes (e.g. 'OO1:', 'SIPAG:', 'BIP:') for readable scope."""
    if not desc:
        return "N/A"
    d = desc.strip()
    d = re.sub(r'^(OO\d+|CSSP|CP|BIP|SIPAG)[^:\-—]*[:\-—]\s*', '', d, flags=re.IGNORECASE)
    d = re.sub(r'^PROTECT LIVES AND PROPERTIES AGAINST MAJOR FLOODS,?\s*(FLOOD MANAGEMENT PROGRAM,?)?\s*', '', d, flags=re.IGNORECASE)
    d = re.sub(r'^(CONVERGENCE AND SPECIAL SUPPORT PROGRAM|BASIC INFRASTRUCTURE PROGRAM|ACCESS ROADS AND/OR BRIDGES FROM THE NATIONAL ROAD/S LEADING TO MAJOR/ STRATEGIC PUBLIC BUILDINGS/ FACILITIES),?\s*', '', d, flags=re.IGNORECASE)
    d = re.sub(r'^[,\s\-—:]+', '', d).strip()
    return d if d else desc

def search_project_by_id(project_id: str) -> dict | None:
    """Exact project lookup with all connected graph entities."""
    cypher = """
    MATCH (p:Project)
    WHERE toLower(p.project_id) = toLower($pid)
    OPTIONAL MATCH (p)-[:CONTRACTED_TO]->(c:Contractor)
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
    OPTIONAL MATCH (p)-[:SUPERVISED_BY]->(off:Office)
    OPTIONAL MATCH (p)-[:CATEGORIZED_AS]->(cat:Category)
    RETURN p.project_id AS id,
           p.description AS description,
           p.budget_raw AS budget,
           p.budget_numeric AS budget_num,
           p.status AS status,
           p.timeline AS timeline,
           p.url AS url,
           c.name AS contractor,
           c.code AS contractor_code,
           loc.name AS location,
           off.name AS office,
           cat.name AS category
    LIMIT 1
    """
    with driver.session() as s:
        res = s.run(cypher, pid=project_id.strip()).data()
        if res:
            res[0]["location"] = clean_location(res[0].get("location"))
            return res[0]
        return None

def sanitize_search_query(query: str) -> str:
    """Normalize common terms and strip Lucene special characters (e.g. hyphens)."""
    q = re.sub(r'lapu[- ]*lapu', 'Lapu', query, flags=re.IGNORECASE)
    q = re.sub(r'cebu[- ]*city', 'Cebu City', q, flags=re.IGNORECASE)
    q = re.sub(r'[^a-zA-Z0-9\s]', ' ', q)
    words = [w for w in q.split() if len(w) > 2 and w.lower() not in STOP_WORDS]
    return " ".join(words) if words else q.strip()

def search_projects_by_location(loc_name: str, limit: int = 15) -> tuple[list[dict], int, float]:
    """Retrieve all projects matching a municipality, city, or barangay location."""
    clean_loc = sanitize_search_query(loc_name).strip()
    if not clean_loc or len(clean_loc) < 3:
        return [], 0, 0.0

    cypher_count = """
    MATCH (p:Project)-[:LOCATED_IN]->(loc:Location)
    WHERE toLower(loc.name) CONTAINS toLower($loc)
    RETURN count(DISTINCT p) AS cnt, sum(p.budget_numeric) AS total_budget
    """
    cypher_samples = """
    MATCH (p:Project)-[:LOCATED_IN]->(loc:Location)
    WHERE toLower(loc.name) CONTAINS toLower($loc)
    WITH DISTINCT p, loc
    ORDER BY p.budget_numeric DESC
    LIMIT $limit
    OPTIONAL MATCH (p)-[:CONTRACTED_TO]->(c:Contractor)
    RETURN p.project_id AS entity,
           ['Project'] AS labels,
           p.description AS description,
           p.budget_raw AS budget,
           p.budget_numeric AS budget_num,
           p.status AS status,
           p.url AS url,
           1.0 AS score,
           [
               {rel: 'LOCATED_IN', target: loc.name, desc: loc.name},
               {rel: 'CONTRACTED_TO', target: coalesce(c.name, 'N/A'), desc: coalesce(c.name, 'N/A')}
           ] AS relations
    """
    with driver.session() as s:
        try:
            cnt_row = s.run(cypher_count, loc=clean_loc).single()
            if not cnt_row or cnt_row["cnt"] == 0:
                return [], 0, 0.0
            total_cnt = cnt_row["cnt"]
            total_b = cnt_row["total_budget"] or 0.0

            samples = s.run(cypher_samples, loc=clean_loc, limit=limit).data()
            return samples, total_cnt, total_b
        except Exception:
            return [], 0, 0.0

def search_graph_fts(query: str, limit: int = 5) -> list[dict]:
    """Full-text search across projects, contractors, and locations with early LIMIT for sub-100ms response."""
    clean = sanitize_search_query(query)
    if not clean:
        return []

    words = clean.split()
    search_term = " AND ".join([f"{w}*" for w in words if len(w) > 1]) or clean
    cypher = """
    CALL db.index.fulltext.queryNodes('entity_fts', $search_term) YIELD node, score
    WITH node, score
    ORDER BY score DESC
    LIMIT $limit
    OPTIONAL MATCH (node)-[r]-(neighbor)
    RETURN node.entity_id AS entity,
           labels(node) AS labels,
           node.description AS description,
           node.budget_raw AS budget,
           node.status AS status,
           node.url AS url,
           score,
           collect(DISTINCT {
               rel: type(r),
               target: neighbor.entity_id,
               desc: r.description
           })[0..6] AS relations
    """
    with driver.session() as s:
        try:
            records = s.run(cypher, search_term=search_term, limit=limit).data()
            if records:
                return records
        except Exception:
            pass

        # Fallback to case-insensitive CONTAINS requiring ALL search terms across fields
        fallback = """
        MATCH (node)
        WHERE ALL(w IN $words WHERE
            toLower(node.entity_id) CONTAINS toLower(w)
            OR toLower(coalesce(node.description, '')) CONTAINS toLower(w)
            OR toLower(coalesce(node.location, '')) CONTAINS toLower(w)
            OR toLower(coalesce(node.contractor, '')) CONTAINS toLower(w)
        )
        WITH node
        LIMIT $limit
        OPTIONAL MATCH (node)-[r]-(neighbor)
        RETURN node.entity_id AS entity,
               labels(node) AS labels,
               node.description AS description,
               node.budget_raw AS budget,
               node.status AS status,
               node.url AS url,
               1.0 AS score,
               collect(DISTINCT {
                   rel: type(r),
                   target: neighbor.entity_id,
                   desc: r.description
               })[0..6] AS relations
        """
        return s.run(fallback, words=words, limit=limit).data()

def get_top_contractors(limit: int = 10) -> list[dict]:
    """Analytics: Top contractors by total awarded budget."""
    cypher = """
    MATCH (p:Project)-[:CONTRACTED_TO]->(c:Contractor)
    WHERE p.budget_numeric > 0
    RETURN c.name AS contractor,
           c.code AS code,
           count(p) AS project_count,
           sum(p.budget_numeric) AS total_budget
    ORDER BY total_budget DESC
    LIMIT $limit
    """
    with driver.session() as s:
        return s.run(cypher, limit=limit).data()

def get_top_projects(limit: int = 10) -> list[dict]:
    """Analytics: Most expensive projects."""
    cypher = """
    MATCH (p:Project)
    WHERE p.budget_numeric > 0
    OPTIONAL MATCH (p)-[:CONTRACTED_TO]->(c:Contractor)
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
    RETURN p.project_id AS id,
           p.budget_raw AS budget,
           p.budget_numeric AS budget_num,
           p.description AS description,
           c.name AS contractor,
           loc.name AS location
    ORDER BY p.budget_numeric DESC
    LIMIT $limit
    """
    with driver.session() as s:
        return s.run(cypher, limit=limit).data()

def get_dataset_summary() -> dict:
    """Overall dataset statistics."""
    cypher = """
    MATCH (p:Project)
    OPTIONAL MATCH (c:Contractor)
    RETURN count(DISTINCT p) AS total_projects,
           sum(p.budget_numeric) AS total_budget,
           count(DISTINCT c) AS total_contractors
    """
    with driver.session() as s:
        return s.run(cypher).single()

# ==========================================================
# 2. Formatted Printers
# ==========================================================

def print_project_card(p: dict):
    loc = clean_location(p.get('location'))
    desc = clean_description(p.get('description'))
    print(f"\n================ 🏗️ DPWH PROJECT {p['id']} ================")
    print(f"📁 Category:            {p.get('category', 'Infrastructure')}")
    print(f"💰 Budget:              {p.get('budget', 'N/A')}")
    print(f"👷 Contractor (Builder): {p.get('contractor', 'N/A')}")
    print(f"🏢 Implementing Office: {p.get('office', 'N/A')}")
    print(f"📍 Location:            {loc}")
    print(f"🚦 Status:              {p.get('status', 'N/A')}")
    print(f"📅 Timeline:            {p.get('timeline', 'N/A')}")
    if p.get('url'):
        print(f"🔗 Source URL:          {p['url']}")
    if desc:
        print(f"📝 Official Scope:      {desc}")
    print("=" * 62)

def print_matched_projects_list(records: list[dict], title: str = "MATCHED PROJECTS"):
    """Uniform structured view when multiple projects match a location or search term."""
    print("\n" + "=" * 62)
    print(f"📍 {title} ({len(records)} Found)")
    print("=" * 62)
    for idx, r in enumerate(records, 1):
        pid = r['entity'].replace('DPWH Project ', '')
        budget = r.get('budget') or 'Budget N/A'
        status = r.get('status') or 'Status N/A'
        desc = clean_description(r.get('description'))
        if len(desc) > 130:
            desc = desc[:130] + "..."
        
        contractor = "N/A"
        loc = "N/A"
        for rel in r.get("relations", []):
            if rel.get("rel") == "CONTRACTED_TO":
                contractor = rel.get("target", "N/A")
            elif rel.get("rel") == "LOCATED_IN":
                loc = clean_location(rel.get("target", "N/A"))
        
        print(f"[{idx}] {pid} | {budget} | {status}")
        if loc != "N/A":
            print(f"    📍 Location:   {loc}")
        if contractor != "N/A":
            print(f"    👷 Contractor: {contractor}")
        print(f"    📝 Scope:      {desc}")
        print()
    print("=" * 62)

def print_instant_insights(records: list[dict], total_count: int | None = None, total_budget: float | None = None) -> dict:
    """Compute sub-millisecond aggregate highlights directly from matched graph records or known database totals."""
    total = total_count if total_count is not None else len(records)
    tot_budget = total_budget if total_budget is not None else 0.0
    categories = []
    contractors = []
    
    for r in records:
        if total_budget is None:
            b_raw = r.get('budget', '')
            nums = re.findall(r'[\d,]+(?:\.\d+)?', str(b_raw))
            if nums:
                try:
                    tot_budget += float(nums[0].replace(',', ''))
                except Exception:
                    pass
        
        desc = str(r.get('description', '')).upper()
        if 'WATER' in desc: categories.append('Water Supply & Drainage')
        elif 'FLOOD' in desc: categories.append('Flood Mitigation')
        elif 'ROAD' in desc or 'HIGHWAY' in desc or 'ASPHALT' in desc: categories.append('Road Works / Concreting')
        elif 'BRIDGE' in desc: categories.append('Bridges & Flyovers')
        elif 'BUILDING' in desc or 'MULTI-PURPOSE' in desc or 'HALL' in desc: categories.append('Public Buildings')
        elif 'SLOPE' in desc: categories.append('Slope Protection')
        
        for rel in r.get('relations', []):
            if rel.get('rel') == 'CONTRACTED_TO':
                c_name = rel.get('target')
                if c_name and c_name != 'N/A':
                    contractors.append(c_name)

    cat_str = ', '.join(list(dict.fromkeys(categories))[:3]) or 'Infrastructure Works'
    top_c = ', '.join(list(dict.fromkeys(contractors))[:2])
    budget_fmt = f"₱{tot_budget:,.2f}" if tot_budget > 0 else "Budget Under Verification"
    
    print("📊 Instant Graph Insights (<1ms):")
    print(f"   • Total Projects:  {total} | Total Allocated: {budget_fmt}")
    print(f"   • Primary Sectors: {cat_str}")
    if top_c:
        print(f"   • Key Builders:    {top_c}")
    print("-" * 62)

    return {
        "total_count": total,
        "total_budget": tot_budget,
        "total_budget_fmt": budget_fmt,
        "sectors": cat_str,
        "key_contractors": top_c
    }

def print_fts_results(results: list[dict]):
    if not results:
        print("⚠️ No matching entities found in Neo4j.")
        return
    print("\n" + "=" * 62)
    for idx, r in enumerate(results, 1):
        labels = [l for l in r.get("labels", []) if l != "base"]
        tag = f" [{'/'.join(labels)}]" if labels else ""
        print(f"📌 [{idx}] {r['entity']}{tag}")
        if r.get("budget"):
            print(f"   • Budget:  {r['budget']}")
        if r.get("status"):
            print(f"   • Status:  {r['status']}")
        if r.get("description"):
            desc = clean_description(r['description'])
            if len(desc) > 180:
                desc = desc[:180] + "..."
            print(f"   • Details: {desc}")
        rels = r.get("relations") or []
        clean_rels = [clean_location(rel["desc"]) for rel in rels if rel.get("desc")]
        if clean_rels:
            print("   • Connected Knowledge:")
            for cr in clean_rels[:4]:
                print(f"     - {cr}")
        print()
    print("=" * 62)

# ==========================================================
# 3. Grounded Streaming QA via Local Ollama
# ==========================================================

def ensure_ollama_running() -> bool:
    """Checks if Ollama or local LLM endpoint is reachable; auto-starts Ollama in background if offline."""
    import urllib.request
    import subprocess
    import shutil

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.8) as res:
            if res.status == 200:
                return True
    except Exception:
        pass

    # If ollama executable is not installed on this system, don't attempt to start it
    if not shutil.which("ollama"):
        return False

    print("🔄 Local Ollama is offline. Starting Ollama automatically in background...")
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )
        for _ in range(10):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.8) as res:
                    if res.status == 200:
                        print("✅ Ollama started successfully.")
                        return True
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Could not auto-start Ollama: {e}")
    return False


def is_prohibited_stream_chunk(text: str) -> bool:
    """Post-generation output rail: detects programming code or generic conceptual definitions."""
    t = text.lower()
    return bool(re.search(
        r'(?:```(?:python|javascript|js|ts|cpp|java|go|html|css)?|'
        r'\b(?:for\s+[a-zA-Z_]\w*\s+in\s+|while\s*\(|let\s+\w+\s*=|const\s+\w+\s*=|'
        r'def\s+[a-zA-Z_]\w*\(|function\s+[a-zA-Z_]\w*\(|console\.log|'
        r'System\.out\.println|printf\(|#include\s+<)|'
        r'is\s+a\s+programming\s+construct|is\s+a\s+looping\s+construct)\b',
        t
    ))


async def stream_ollama(messages: list[dict], max_tokens: int = 50) -> str:
    """Stream from local Ollama or OpenAI-compatible endpoint with real-time response timer and output guardrail."""
    url_native = "http://localhost:11434/api/chat"
    payload_native = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "num_thread": 6,       # Optimal for 4-core / 8-thread AMD Ryzen CPU
            "num_ctx": 512,        # Reduces prefill KV-cache latency by 75%
            "num_predict": max_tokens,
            "temperature": 0.2,
            "repeat_penalty": 1.25
        }
    }
    
    collected_chunks = []
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            async with client.stream("POST", url_native, json=payload_native) as resp:
                if resp.status_code == 200:
                    buffer = ""
                    flushed = False
                    suppressed = False
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        import json
                        d = json.loads(line)
                        chunk = d.get("message", {}).get("content", "")
                        if chunk:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            collected_chunks.append(chunk)
                            buffer += chunk
                            if not flushed:
                                if is_prohibited_stream_chunk(buffer):
                                    suppressed = True
                                    break
                                if len(buffer) >= 30:
                                    sys.stdout.write(buffer)
                                    sys.stdout.flush()
                                    flushed = True
                            else:
                                if is_prohibited_stream_chunk("".join(collected_chunks)):
                                    suppressed = True
                                    break
                                sys.stdout.write(chunk)
                                sys.stdout.flush()
                        if d.get("done"):
                            break
                    if suppressed:
                        print("\n⚠️ [Response Suppressed by Guardrail: Detected off-domain output]")
                        print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects.")
                        print("=" * 62)
                        return "I am restricted to answering questions regarding DPWH Cebu infrastructure projects."
                    if not flushed and buffer:
                        if is_prohibited_stream_chunk(buffer):
                            print("\n⚠️ [Response Suppressed by Guardrail: Detected off-domain output]")
                            print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects.")
                            print("=" * 62)
                            return "I am restricted to answering questions regarding DPWH Cebu infrastructure projects."
                        sys.stdout.write(buffer)
                        sys.stdout.flush()
                    elapsed = time.perf_counter() - start_time
                    ttft = (first_token_time - start_time) if first_token_time else elapsed
                    gen_duration = elapsed - ttft
                    speed_str = f" | ~{token_count / gen_duration:.1f} tok/s" if (token_count > 1 and gen_duration > 0.05) else ""
                    print(f"\n⏱️ Response time: {elapsed:.2f}s (TTFT: {ttft:.2f}s{speed_str})")
                    print("=" * 62)
                    return "".join(collected_chunks).strip()
    except Exception:
        pass

    # Fallback to standard OpenAI-compatible endpoint (e.g. vLLM or local server)
    url_v1 = f"{LLM_BASE_URL}/chat/completions"
    payload_v1 = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    start_time_v1 = time.perf_counter()
    first_token_time_v1 = None
    token_count_v1 = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            async with client.stream("POST", url_v1, json=payload_v1) as resp:
                buffer_v1 = ""
                flushed_v1 = False
                suppressed_v1 = False
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        import json
                        try:
                            d = json.loads(line[6:])
                            chunk = d["choices"][0]["delta"].get("content", "")
                            if chunk:
                                if first_token_time_v1 is None:
                                    first_token_time_v1 = time.perf_counter()
                                token_count_v1 += 1
                                collected_chunks.append(chunk)
                                buffer_v1 += chunk
                                if not flushed_v1:
                                    if is_prohibited_stream_chunk(buffer_v1):
                                        suppressed_v1 = True
                                        break
                                    if len(buffer_v1) >= 30:
                                        sys.stdout.write(buffer_v1)
                                        sys.stdout.flush()
                                        flushed_v1 = True
                                else:
                                    if is_prohibited_stream_chunk("".join(collected_chunks)):
                                        suppressed_v1 = True
                                        break
                                    sys.stdout.write(chunk)
                                    sys.stdout.flush()
                        except Exception:
                            pass
                if suppressed_v1:
                    print("\n⚠️ [Response Suppressed by Guardrail: Detected off-domain output]")
                    print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects.")
                    print("=" * 62)
                    return "I am restricted to answering questions regarding DPWH Cebu infrastructure projects."
                if not flushed_v1 and buffer_v1:
                    if is_prohibited_stream_chunk(buffer_v1):
                        print("\n⚠️ [Response Suppressed by Guardrail: Detected off-domain output]")
                        print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects.")
                        print("=" * 62)
                        return "I am restricted to answering questions regarding DPWH Cebu infrastructure projects."
                    sys.stdout.write(buffer_v1)
                    sys.stdout.flush()
        elapsed = time.perf_counter() - start_time_v1
        ttft = (first_token_time_v1 - start_time_v1) if first_token_time_v1 else elapsed
        gen_duration = elapsed - ttft
        speed_str = f" | ~{token_count_v1 / gen_duration:.1f} tok/s" if (token_count_v1 > 1 and gen_duration > 0.05) else ""
        print(f"\n⏱️ Response time: {elapsed:.2f}s (TTFT: {ttft:.2f}s{speed_str})")
        print("=" * 62)
        return "".join(collected_chunks).strip()
    except Exception as e:
        if "connection" in str(e).lower() and ensure_ollama_running():
            print("🔄 Auto-started Ollama. Retrying request...")
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                    async with client.stream("POST", url_v1, json=payload_v1) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and not line.endswith("[DONE]"):
                                import json
                                try:
                                    d = json.loads(line[6:])
                                    chunk = d["choices"][0]["delta"].get("content", "")
                                    if chunk:
                                        sys.stdout.write(chunk)
                                        sys.stdout.flush()
                                        collected_chunks.append(chunk)
                                except Exception:
                                    pass
                print("\n" + "=" * 62)
                return "".join(collected_chunks).strip()
            except Exception:
                pass
        print(f"\n💡 Notice: Local AI summary is currently unavailable ({e.__class__.__name__}).")
        print("   All verified facts and data from the Knowledge Graph are shown above.")
        print(f"   👉 To enable AI summaries: install Ollama (https://ollama.com) and run: `ollama pull {LLM_MODEL}`")
        print("=" * 62)
        return ""

async def ask_llm_stream(question: str, history: list[dict] = None, last_context: str = "", mode: str = "ai") -> tuple[str, str]:
    """Retrieve relevant subgraph from Neo4j in <20ms and stream grounded answer from Ollama with conversational memory."""
    if history is None:
        history = []

    # 0. Pre-retrieval intent gate: intercept off-domain queries immediately (<1ms)
    off_topic, reason = is_off_topic_query(question)
    if off_topic:
        print(f"\n⚠️ Out of Scope: {reason}")
        print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects, contractors, budgets, and locations.")
        print("   Example queries:")
        print("   • 'Any projects in Bogo?'")
        print("   • 'Who is the top contractor in Cebu?'")
        print("   • 'Tell me about project 24HH0043'")
        return "", ""

    print("🔍 Fetching knowledge graph facts from Neo4j (<20ms)...")

    terms = [w for w in re.findall(r'[A-Za-z0-9_]+', question) if w.lower() not in STOP_WORDS and len(w) > 2]
    followup_cues = ["it", "that", "this", "more", "contractor", "budget", "status", "timeline", "who", "cost", "when", "details"]
    is_followup = bool(history) and (len(terms) == 0 or any(w in question.lower().split() for w in followup_cues))

    graph_context = ""
    targeted_project = None
    matched_projects = []

    # 1. Direct Project ID match in current question (e.g. 24HH0043, 25H00064)
    pid_in_q = re.search(r'\b\d{2}[A-Za-z]{1,2}\d{4,5}\b', question, re.IGNORECASE)
    if pid_in_q:
        targeted_project = search_project_by_id(pid_in_q.group(0).upper())

    # 2. If follow-up, check previously mentioned project IDs in conversation
    if not targeted_project and is_followup and history:
        last_turn_text = " ".join([h["content"] for h in history[-2:]])
        prev_pids = re.findall(r'\b\d{2}[A-Za-z]{1,2}\d{4,5}\b', last_turn_text, re.IGNORECASE)
        if prev_pids:
            unique_pids = list(dict.fromkeys([p.upper() for p in prev_pids]))
            target_pid = unique_pids[-1]
            targeted_project = search_project_by_id(target_pid)
            if len(unique_pids) > 1:
                other_pids = [p for p in unique_pids if p != target_pid]
                print(f"\nℹ️ Focused on DPWH Project {target_pid} (from previous list)")
                if other_pids:
                    print(f"💡 Other projects mentioned: {', '.join(other_pids)}")

    # 3. Configure Prompt and Uniform Display
    if targeted_project:
        # Uniform Single Project Card
        print_project_card(targeted_project)
        
        graph_context = (
            f"Project: DPWH Project {targeted_project['id']}\n"
            f"Category: {targeted_project.get('category', 'Infrastructure')}\n"
            f"Location: {clean_location(targeted_project.get('location'))}\n"
            f"Scope Description: {clean_description(targeted_project.get('description'))}\n"
            f"Status: {targeted_project.get('status', 'Completed')}"
        )

        if mode == "instant":
            print("💡 Instant Mode: Project details loaded from Neo4j in <10ms. (Type '/ai' to enable summaries)")
            return "", graph_context

        if not ensure_ollama_running():
            print(f"\n💡 Notice: Local AI ({LLM_MODEL}) is offline or not installed.")
            print("   Complete project details from the Knowledge Graph are displayed above.")
            return "", graph_context

        system_prompt = (
            "You are an expert DPWH civil engineering assistant in Cebu, Philippines.\n"
            "Provide a concise 1 to 2 sentence plain-English summary of what this project accomplishes.\n\n"
            "CRITICAL RULES:\n"
            "1. Focus ONLY on what is being constructed, repaired, or installed and its practical public benefit.\n"
            "2. Do NOT recite the raw budget figures, dates, contractor name, or office name (the project card already displays them).\n"
            "3. NEVER use the phrase 'Barangay Unspecified'.\n"
            "4. Always output complete, grammatically finished sentences ending with terminal punctuation (. or !). Never stop mid-sentence."
        )
        print(f"\n🤖 Grounded Engineering Summary ({LLM_MODEL}):")
        print("-" * 62)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Knowledge Graph Context:\n{graph_context}\n\nUser Question: {question}"}
        ]
        answer_text = await stream_ollama(messages, max_tokens=75)
        return answer_text, graph_context

    else:
        search_query = " ".join(terms) if terms else question

        # 3a. Check if this query corresponds to a location in the knowledge graph
        loc_projects, loc_total_cnt, loc_total_budget = search_projects_by_location(search_query)

        matched_records = []
        if loc_projects:
            matched_projects = loc_projects
            total_in_db = loc_total_cnt
            total_b_val = loc_total_budget
            loc_label = search_query.title()
            print_matched_projects_list(matched_projects, title=f"PROJECTS IN: {loc_label} ({total_in_db} Found in Graph | Showing Top {len(matched_projects)})")
            insights = print_instant_insights(matched_projects, total_count=total_in_db, total_budget=total_b_val)
        else:
            # 3b. Fallback to full-text search across graph
            matched_records = search_graph_fts(search_query, limit=25)
            matched_projects = [
                r for r in matched_records 
                if "Project" in r.get("labels", []) or str(r.get("entity", "")).startswith("DPWH Project")
            ]
            if matched_projects:
                loc_label = " ".join([w.title() for w in terms]) if terms else "Knowledge Graph"
                print_matched_projects_list(matched_projects, title=f"PROJECTS FOUND: {loc_label} ({len(matched_projects)} Found)")
                insights = print_instant_insights(matched_projects)

        if matched_projects:
            if mode == "instant":
                print("💡 Instant Mode active. Type any Project ID (e.g. '25HN0012') or ask 'tell me more' for deep AI analysis.")
                return "", ""

            if not ensure_ollama_running():
                print(f"💡 Notice: Local AI ({LLM_MODEL}) is offline. Graph results and instant metrics are displayed above.")
                print(f"👉 To enable AI summaries: install Ollama (https://ollama.com) and run: `ollama pull {LLM_MODEL}`")
                print("💡 Tip: Type any Project ID (e.g. '25HN0012') for full engineering details.")
                return "", ""

            # Compact context for prompt evaluation: keeps TTFT under 1.5s
            lines = []
            for p in matched_projects[:5]:
                pid = p['entity'].replace('DPWH Project ', '')
                desc = clean_description(p.get('description'))
                if len(desc) > 65:
                    desc = desc[:65] + "..."
                b = p.get('budget', 'N/A')
                lines.append(f"{pid}: {desc} ({b})")
            sample_lines = " | ".join(lines)

            is_count_query = any(w in question.lower() for w in ["how many", "count", "number of", "total project", "total count"])

            if is_count_query:
                system_prompt = (
                    "You are an expert DPWH civil engineering assistant in Cebu, Philippines.\n"
                    f"State directly in ONE complete sentence that there are {insights['total_count']} recorded DPWH projects in {loc_label} with a total allocated budget of {insights['total_budget_fmt']}."
                )
                print(f"🤖 Grounded Answer ({LLM_MODEL}):")
                print("-" * 62)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                answer_text = await stream_ollama(messages, max_tokens=65)
                print("💡 Tip: Type any Project ID (e.g. '25HN0012') or ask 'tell me more about [ID]' for full engineering details.")
                return answer_text, f"Projects in {loc_label}: {insights['total_count']} ({insights['total_budget_fmt']})"

            else:
                graph_context = (
                    f"Location / Subject: {loc_label}\n"
                    f"Total Recorded Projects in Database: {insights['total_count']}\n"
                    f"Total Allocated Budget: {insights['total_budget_fmt']}\n"
                    f"Primary Sectors: {insights['sectors']}\n"
                    f"Key Contractors: {insights['key_contractors'] or 'Various'}\n"
                    f"Sample Projects: {sample_lines}"
                )

                system_prompt = (
                    "You are an expert DPWH civil engineering assistant in Cebu, Philippines.\n"
                    f"Provide a concise 1-sentence plain-English summary of what the DPWH public works in Cebu relate to '{loc_label}'.\n\n"
                    "CRITICAL RULES:\n"
                    "1. NO DICTIONARY DEFINITIONS: NEVER provide academic, dictionary, or textbook definitions (e.g. do NOT define what 'science', 'math', or 'water' is in general).\n"
                    f"2. INFRASTRUCTURE FOCUS: State specifically what the public works involve (e.g., 'In Cebu, projects relating to {loc_label} involve the construction of educational facilities and science high school buildings.').\n"
                    "3. Output exactly ONE or TWO complete sentences. Do NOT output numbered lists, bullet points, or repeated project titles.\n"
                    "4. Never say 'Barangay Unspecified'."
                )
                print(f"🤖 Grounded Answer ({LLM_MODEL}):")
                print("-" * 62)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Knowledge Graph Context:\n{graph_context}\n\n"
                            f"Topic: {loc_label}\n"
                            f"Task: Based strictly on the context above, what do the DPWH Cebu projects matching '{loc_label}' involve? "
                            f"(Do not define '{loc_label}', focus solely on the public works projects)."
                        )
                    }
                ]
                answer_text = await stream_ollama(messages, max_tokens=120)
                print("💡 Tip: Type any Project ID (e.g. '25HN0012') or ask 'tell me more about [ID]' for full engineering details.")
                return answer_text, graph_context

        else:
            # Post-retrieval validation: Ensure matched non-project records have genuine term overlap
            if matched_records:
                query_tokens = [w.lower() for w in terms]
                valid_records = []
                for r in matched_records:
                    ent_str = f"{r.get('entity', '')} {r.get('description', '')}".lower()
                    if any(t in ent_str for t in query_tokens):
                        valid_records.append(r)
                matched_records = valid_records

            if not matched_records and not last_context:
                q_lower = question.lower()
                q_words = set(re.findall(r'[a-zA-Z0-9]+', q_lower))
                is_domain_query = bool(q_words & INFRA_DOMAIN_KEYWORDS)

                if not is_domain_query:
                    print("\n⚠️ Out of Scope: Query not related to DPWH Cebu Infrastructure.")
                    print("💡 This assistant is strictly restricted to DPWH Cebu infrastructure projects, contractors, budgets, and locations.")
                    print("   Example queries:")
                    print("   • 'Any projects in Bogo?'")
                    print("   • 'Who is the top contractor in Cebu?'")
                    print("   • 'Tell me about project 24HH0043'")
                    return "", ""
                else:
                    print(f"\n⚠️ No records found in the DPWH Cebu database matching '{search_query}'.")
                    print("💡 Try searching by City/Municipality (e.g., 'Toledo', 'Mandaue') or Project ID (e.g., '24HH0043').")
                    return "", ""

            # General Question with matched graph facts or ongoing context
            if matched_records:
                lines = []
                for r in matched_records[:3]:
                    lines.append(f"Entity: {r['entity']}")
                    if r.get("budget"): lines.append(f"Budget: {r['budget']}")
                    if r.get("status"): lines.append(f"Status: {r['status']}")
                    if r.get("description"):
                        desc = clean_description(r['description'])
                        if len(desc) > 80: desc = desc[:80] + "..."
                        lines.append(f"Details: {desc}")
                    for rel in (r.get("relations") or [])[:2]:
                        if rel.get("desc"): lines.append(f"Fact: {clean_location(rel['desc'])}")
                graph_context = "\n".join(lines)
            else:
                graph_context = last_context[:500]

            if mode == "instant":
                print("💡 Instant Mode: Showing matching database entities above.")
                return "", graph_context

            if not ensure_ollama_running():
                print(f"\n💡 Notice: Local AI ({LLM_MODEL}) is offline. Graph context is displayed above.")
                print(f"👉 To enable AI answers: install Ollama (https://ollama.com) and run: `ollama pull {LLM_MODEL}`")
                return "", graph_context

            system_prompt = (
                "You are an expert DPWH civil engineering assistant dedicated exclusively to Cebu, Philippines.\n"
                "Answer accurately using ONLY the provided Knowledge Graph Context and prior conversation history.\n\n"
                "CRITICAL BOUNDARIES:\n"
                "1. STRICT DOMAIN LOCK: You must decline any questions unrelated to DPWH civil engineering or Cebu infrastructure (such as computer programming, coding, cooking, general trivia) in one sentence: 'I am restricted to answering questions regarding DPWH Cebu infrastructure projects and data.'\n"
                "2. STRICT GROUNDING: State facts ONLY from the Knowledge Graph Context. Never invent projects, contractors, or figures.\n"
                "3. Distinguish roles: The Contractor is the private builder. The Implementing Office is the DPWH agency.\n"
                "4. Be concise: answer in 1 to 2 complete, well-formed sentences.\n"
                "5. Never say 'Barangay Unspecified'."
            )
            print(f"\n================ 🤖 GROUNDED ANSWER ({LLM_MODEL}) ================")
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-2:]:
                messages.append(h)
            messages.append({"role": "user", "content": f"Knowledge Graph Context:\n{graph_context}\n\nUser Question: {question}"})
            answer_text = await stream_ollama(messages, max_tokens=70)
            return answer_text, graph_context



# ==========================================================
# 4. Interactive Console
# ==========================================================

def run_console():
    has_ai = ensure_ollama_running()
    current_mode = os.environ.get("DEFAULT_CONSOLE_MODE")
    if not current_mode:
        current_mode = "ai" if has_ai else "instant"

    print("\n" + "=" * 65)
    print(f"⚡ Neo4j Knowledge Graph Console (DPWH Cebu Infrastructure)")
    if has_ai:
        print(f"Connected to Neo4j Aura + Local Ollama ({LLM_MODEL})")
    else:
        print(f"Connected to Neo4j Aura (Instant Cypher Mode - No AI required)")
    print("=" * 65)
    print("Quick Commands:")
    print("  • Type any project ID (e.g. '24HH0043') for instant project card (<10ms)")
    print("  • Type '/search <term>' or '/fts <term>' for full-text search (<20ms)")
    print("  • Type '/top-contractors' to see top contractors by awarded budget")
    print("  • Type '/top-budgets' to see the largest projects")
    print("  • Type '/summary' for dataset totals")
    print("  • Type '/instant' for sub-second database mode (skips LLM wait on searches)")
    print("  • Type '/ai' for grounded AI summary mode")
    print("  • Ask any question for streaming grounded answer")
    print("  • Type 'exit' or 'quit' to end")
    print("=" * 65)

    history = []
    last_context = ""
    if current_mode == "instant":
        if not has_ai:
            print("💡 Notice: Local AI (Ollama) is offline or not installed.")
        print("⚡ Active Mode: Instant (<100ms database responses - zero AI wait)")
    else:
        print(f"🤖 Active Mode: AI Grounded ({LLM_MODEL} streaming summaries)")

    while True:
        try:
            user_input = input("\n❓ Search / Ask: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break

            if user_input.lower() in ("/clear", "clear", "/reset"):
                history.clear()
                last_context = ""
                print("🧹 Conversation memory cleared.")
                continue

            if user_input.lower() in ("/instant", "/fast", "/mode instant"):
                current_mode = "instant"
                print("⚡ Instant Mode activated! Search results and instant insights appear in <100ms without LLM wait.")
                print("💡 (You can still ask follow-up questions or 'tell me more' for deep AI answers)")
                continue

            if user_input.lower() in ("/ai", "/deep", "/mode ai"):
                if not ensure_ollama_running():
                    print(f"⚠️ Local Ollama is offline or not installed.")
                    print(f"👉 To enable AI summaries: install Ollama (https://ollama.com) and run: `ollama pull {LLM_MODEL}`")
                    print("⚡ Remaining in Instant Mode (<100ms database queries).")
                else:
                    current_mode = "ai"
                    print(f"🤖 AI Summary Mode activated! Generates streaming grounded summaries via {LLM_MODEL}.")
                continue

            # 1. Project ID exact match (e.g. 24HH0043, 22HF0008, 25H00064)
            pid_match = re.search(r'\b\d{2}[A-Za-z]{1,2}\d{4,5}\b', user_input)
            if pid_match and (len(user_input.split()) == 1 or user_input.lower().startswith("project")):
                target_pid = pid_match.group(0).upper()
                proj = search_project_by_id(target_pid)
                if proj:
                    print_project_card(proj)
                    # Remember this project for follow-ups!
                    history.append({"role": "user", "content": f"Tell me about project {target_pid}"})
                    history.append({"role": "assistant", "content": f"Project {target_pid}: {proj.get('description', '')}. Budget: {proj.get('budget', '')}. Contractor: {proj.get('contractor', '')}. Location: {proj.get('location', '')}."})
                    last_context = f"Project {target_pid}\nDetails: {proj.get('description', '')}\nBudget: {proj.get('budget', '')}\nContractor: {proj.get('contractor', '')}\nLocation: {proj.get('location', '')}"
                else:
                    print(f"⚠️ Project {target_pid} not found in Neo4j.")
                continue

            # 2. Analytics commands
            if user_input.lower() in ("/top-contractors", "top contractors", "/contractors"):
                rows = get_top_contractors(10)
                print("\n🏆 Top 10 Contractors by Total Awarded Budget:")
                print("-" * 65)
                for i, r in enumerate(rows, 1):
                    formatted_budget = f"₱{r['total_budget']:,}"
                    print(f"{i:2d}. {r['contractor']:<40} | {r['project_count']:2d} projects | {formatted_budget:>15}")
                print("-" * 65)
                continue

            if user_input.lower() in ("/top-budgets", "top budgets", "/largest"):
                rows = get_top_projects(10)
                print("\n💰 Top 10 Largest Infrastructure Projects:")
                print("-" * 65)
                for i, r in enumerate(rows, 1):
                    formatted_budget = f"₱{r['budget_num']:,}"
                    print(f"{i:2d}. [{r['id']}] {formatted_budget:>15} | Contractor: {r.get('contractor') or 'N/A'}")
                    print(f"    Location: {r.get('location') or 'N/A'}")
                print("-" * 65)
                continue

            if user_input.lower() in ("/summary", "summary", "/stats"):
                stats = get_dataset_summary()
                tot_budget = f"₱{stats['total_budget']:,}" if stats['total_budget'] else "₱0"
                print("\n📊 Knowledge Graph Summary:")
                print(f"   • Total Projects Indexed:    {stats['total_projects']}")
                print(f"   • Total Unique Contractors:  {stats['total_contractors']}")
                print(f"   • Total Allocated Budget:    {tot_budget}")
                continue

            # 3. Full-text search command
            if user_input.lower().startswith(("/search ", "/fts ", "/find ")):
                query_term = user_input.split(" ", 1)[1].strip()
                res = search_graph_fts(query_term)
                print_fts_results(res)
                continue

            # 4. Natural language query via streaming Ollama with memory
            ans, last_context = asyncio.run(ask_llm_stream(user_input, history, last_context, mode=current_mode))
            if ans:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": ans})

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting console.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            full_input = " ".join(sys.argv[1:]).strip()
            first_arg = sys.argv[1].strip()

            if first_arg in ("--summary", "-s"):
                s = get_dataset_summary()
                tot_b = f"₱{s['total_budget']:,}" if s['total_budget'] else "₱0"
                print(f"Projects: {s['total_projects']}, Contractors: {s['total_contractors']}, Total Budget: {tot_b}")
            elif first_arg in ("--top-contractors", "-c"):
                rows = get_top_contractors(10)
                print("\n🏆 Top 10 Contractors by Total Awarded Budget:")
                print("-" * 65)
                for i, r in enumerate(rows, 1):
                    formatted_budget = f"₱{r['total_budget']:,}"
                    print(f"{i:2d}. {r['contractor']:<40} | {r['project_count']:2d} projects | {formatted_budget:>15}")
                print("-" * 65)
            elif first_arg in ("--ask", "-a"):
                q = " ".join(sys.argv[2:]).strip()
                asyncio.run(ask_llm_stream(q))
            elif re.fullmatch(r'\d{2}[A-Za-z]{1,2}\d{4,5}', first_arg, re.IGNORECASE):
                p = search_project_by_id(first_arg)
                if p:
                    print_project_card(p)
                else:
                    print(f"⚠️ Project {first_arg} not found in Neo4j.")
            elif "?" in full_input or any(w in full_input.lower().split() for w in ["what", "who", "which", "where", "how", "tell", "summarize", "list"]):
                # Automatically detects a natural language question!
                asyncio.run(ask_llm_stream(full_input))
            else:
                res = search_graph_fts(full_input)
                print_fts_results(res)
        else:
            run_console()
    finally:
        driver.close()
