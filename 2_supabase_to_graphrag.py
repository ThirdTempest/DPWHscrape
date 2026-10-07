import os
import sys
import re
import json
import shutil
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from neo4j import GraphDatabase

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Suppress internal library logging noise
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

load_dotenv()

# --- 1. Supabase Credentials ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- 2. Neo4j Graph Database Credentials ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- 3. Local LLM Configuration (Ollama) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

# --- 4. Local Embedding Model Configuration (Ollama nomic-embed-text) ---
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Validation
required_vars = [
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
    ("NEO4J_URI", NEO4J_URI),
    ("NEO4J_PASSWORD", NEO4J_PASSWORD),
]

missing = [name for name, val in required_vars if not val]
if missing:
    raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Embedding Client (pointing to local Ollama)
embed_client = AsyncOpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
)

WORKING_DIR = "./dpwh_knowledge_graph"

# Flag to manually trigger a full reset if desired (via --reset CLI flag or RESET_GRAPH=true in .env)
RESET_GRAPH = os.getenv("RESET_GRAPH", "false").lower() in ("true", "1", "yes") or "--reset" in sys.argv

# Clean out working directory only if explicitly requested, or if only failed/broken files exist
has_only_failed_cache = False
if os.path.exists(WORKING_DIR):
    status_file = os.path.join(WORKING_DIR, "kv_store_doc_status.json")
    vdb_rel_file = os.path.join(WORKING_DIR, "vdb_relationships.json")
    if os.path.exists(status_file) and not os.path.exists(vdb_rel_file):
        has_only_failed_cache = True

if RESET_GRAPH or has_only_failed_cache:
    print("🧹 Cleaning out graph cache and resetting sync status...")
    shutil.rmtree(WORKING_DIR, ignore_errors=True)
    supabase.table("scraped_pages").update({"graph_synced": False}).neq("url", "").execute()

os.makedirs(WORKING_DIR, exist_ok=True)

# 1. Local LLM Wrapper (Ollama)
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Call local Ollama model via OpenAI-compatible endpoint."""
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )

# 2. Local Embedding Wrapper (Ollama nomic-embed-text)
async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    """Generate dense embeddings using local Ollama nomic-embed-text."""
    batch_size = 25
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = await embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for item in res.data:
            v = np.array(item.embedding, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            all_vectors.append(v)

    return np.array(all_vectors, dtype=np.float32)

# 3. Initialize LightRAG with Neo4j as Graph Storage & Local Ollama as LLM
rag = LightRAG(
    working_dir=WORKING_DIR,
    graph_storage="Neo4JStorage",
    llm_model_name=LLM_MODEL,
    llm_model_func=llm_model_func,
    llm_model_max_async=1,  # Single-worker for CPU to avoid thread contention
    embedding_func_max_async=2,
    embedding_batch_num=25,
    default_embedding_timeout=180,
    default_llm_timeout=600,  # Generous timeout for CPU generation
    entity_extract_max_gleaning=0,  # Skip redundant passes for maximum speed
    embedding_func=EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=2048,
        func=embedding_func,
    ),
)

async def sanitize_and_sync_cache():
    """Clean up any stuck/failed items in doc status kv store and ensure Supabase reflects processed docs."""
    status_file = os.path.join(WORKING_DIR, "kv_store_doc_status.json")
    if not os.path.exists(status_file):
        return

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            doc_statuses = json.load(f)

        # 1. Clean out failed or processing items so they can be re-indexed cleanly
        cleaned = {
            k: v for k, v in doc_statuses.items()
            if v.get("status") == "processed"
        }
        if len(cleaned) < len(doc_statuses):
            print(f"🧹 Cleaned {len(doc_statuses) - len(cleaned)} failed/stuck doc(s) from local cache for retry.")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False)

        # 2. Mark all successfully processed documents as graph_synced: True in Supabase
        processed_urls = []
        for d in cleaned.values():
            if "SOURCE_URL:" in d.get("content_summary", ""):
                match = re.search(r"SOURCE_URL: (\S+)", d["content_summary"])
                if match:
                    processed_urls.append(match.group(1))

        if processed_urls:
            for url in processed_urls:
                supabase.table("scraped_pages").update({"graph_synced": True}).eq("url", url).execute()

        # 3. Prune orphaned text chunks in KV store to stay in sync with vdb_chunks
        vdb_chunks_file = os.path.join(WORKING_DIR, "vdb_chunks.json")
        kv_chunks_file = os.path.join(WORKING_DIR, "kv_store_text_chunks.json")
        if os.path.exists(vdb_chunks_file) and os.path.exists(kv_chunks_file):
            try:
                with open(vdb_chunks_file, "r", encoding="utf-8") as f:
                    vdb_data = json.load(f)
                vdb_chunk_ids = {d["__id__"] for d in vdb_data.get("data", [])}
                with open(kv_chunks_file, "r", encoding="utf-8") as f:
                    kv_chunks = json.load(f)
                cleaned_chunks = {k: v for k, v in kv_chunks.items() if k in vdb_chunk_ids}
                if len(cleaned_chunks) < len(kv_chunks):
                    with open(kv_chunks_file, "w", encoding="utf-8") as f:
                        json.dump(cleaned_chunks, f, ensure_ascii=False)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Cache maintenance notice: {e}")

from sync_to_neo4j import sync_projects_to_neo4j

async def sync_supabase_to_graph(limit: int = None):
    """Fast deterministic sync from Supabase into Neo4j Knowledge Graph (<3 seconds)."""
    sync_projects_to_neo4j(limit=limit)

def init_neo4j_fts():
    """Ensure Neo4j Full-Text Search (FTS) index exists for sub-100ms entity lookups."""
    try:
        with neo4j_driver.session() as session:
            session.run("CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS FOR (n:base) ON EACH [n.entity_id, n.description]")
    except Exception as e:
        logger.warning(f"Neo4j FTS index init notice: {e}")

def fast_graph_fts_search(query: str, limit: int = 5) -> list[dict]:
    """Zero-LLM instant Full-Text & Graph Traversal Search (<50ms latency)."""
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', query).strip()
    if not cleaned:
        return []

    search_term = " OR ".join([f"{w}*" for w in cleaned.split() if len(w) > 1]) or cleaned
    cypher_fts = """
        CALL db.index.fulltext.queryNodes('entity_fts', $search_term) YIELD node, score
        OPTIONAL MATCH (node)-[r]-(neighbor)
        RETURN node.entity_id AS entity,
               node.description AS description,
               node.entity_type AS entity_type,
               score,
               collect(DISTINCT {
                   rel: type(r),
                   target: neighbor.entity_id,
                   desc: r.description
               })[0..8] AS relations
        ORDER BY score DESC
        LIMIT $limit
    """

    fallback_cypher = """
        MATCH (node)
        WHERE toLower(node.entity_id) CONTAINS toLower($term)
           OR toLower(node.description) CONTAINS toLower($term)
        OPTIONAL MATCH (node)-[r]-(neighbor)
        RETURN node.entity_id AS entity,
               node.description AS description,
               node.entity_type AS entity_type,
               1.0 AS score,
               collect(DISTINCT {
                   rel: type(r),
                   target: neighbor.entity_id,
                   desc: r.description
               })[0..8] AS relations
        LIMIT $limit
    """

    with neo4j_driver.session() as session:
        try:
            records = session.run(cypher_fts, search_term=search_term, limit=limit).data()
            if records:
                return records
        except Exception:
            pass
        return session.run(fallback_cypher, term=cleaned, limit=limit).data()

def format_fts_results(results: list[dict]) -> str:
    """Format structured Neo4j FTS search results cleanly."""
    if not results:
        return "⚠️ No matching entities found in Neo4j knowledge graph."
    lines = []
    for idx, r in enumerate(results, 1):
        score_info = f" (Score: {r['score']:.2f})" if "score" in r and r["score"] is not None else ""
        lines.append(f"📌 [{idx}] {r['entity']}{score_info}")
        if r.get("description"):
            lines.append(f"   • Details: {r['description']}")
        rels = r.get("relations") or []
        clean_rels = [rel["desc"] for rel in rels if rel.get("desc")]
        if clean_rels:
            lines.append("   • Connected Knowledge:")
            for cr in clean_rels:
                lines.append(f"     - {cr}")
        lines.append("")
    return "\n".join(lines).strip()

def extract_keywords_fast(query: str) -> tuple[list[str], list[str]]:
    """Heuristic keyword extraction to bypass the 45-second LLM keyword extraction call."""
    codes = re.findall(r'\b\d{2}[A-Za-z]{2}\d{4}\b', query, re.IGNORECASE)
    quoted = [q[0] or q[1] for q in re.findall(r'"([^"]+)"|\'([^\']+)\'', query)]
    stop_words = {
        "what", "is", "the", "for", "in", "of", "and", "a", "an", "who", "which", "where",
        "how", "tell", "me", "about", "project", "details", "info", "give", "list", "show"
    }
    words = [w for w in re.findall(r'\b[A-Za-z0-9_]+\b', query) if w.lower() not in stop_words and len(w) > 2]
    hl = list(dict.fromkeys(codes + quoted + words))
    ll = list(dict.fromkeys(codes + words))
    return hl or [query], ll or [query]

async def ask_graphrag(query: str, mode: str = "local"):
    """Query GraphRAG with pre-extracted keywords and real-time token streaming."""
    hl, ll = extract_keywords_fast(query)
    param = QueryParam(
        mode=mode,
        hl_keywords=hl,
        ll_keywords=ll,
        enable_rerank=False,
        response_type="Concise Summary",
        stream=True,
    )
    result = await rag.aquery(query, param=param)
    if hasattr(result, "__aiter__"):
        full_text = []
        async for chunk in result:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_text.append(chunk)
        print()
        return "".join(full_text)
    else:
        print(result)
        return result

async def main():
    await sanitize_and_sync_cache()
    await rag.initialize_storages()
    await initialize_pipeline_status()
    init_neo4j_fts()

    # CLI Flag Handling
    sync_requested = "--sync" in sys.argv
    fts_requested = "--fts" in sys.argv
    context_requested = "--context" in sys.argv
    limit = None
    args_to_skip = set()

    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[i + 1])
                args_to_skip.add(i)
                args_to_skip.add(i + 1)
            except ValueError:
                pass
        elif arg in ("--sync", "--fts", "--context", "--reset"):
            args_to_skip.add(i)

    remaining_args = [
        arg for i, arg in enumerate(sys.argv[1:], start=1)
        if i not in args_to_skip and not arg.startswith("--")
    ]
    cli_question = " ".join(remaining_args).strip()

    if sync_requested:
        await sync_supabase_to_graph(limit=limit)
        if not cli_question:
            return

    # Direct FTS search CLI flag
    if fts_requested and cli_question:
        print(f"\n⚡ [Fast FTS Search] Querying Neo4j for '{cli_question}' (<50ms)...")
        results = fast_graph_fts_search(cli_question)
        print("\n" + "=" * 60)
        print(format_fts_results(results))
        print("=" * 60 + "\n")
        return

    # Raw context retrieval CLI flag
    if context_requested and cli_question:
        hl, ll = extract_keywords_fast(cli_question)
        param = QueryParam(mode="local", hl_keywords=hl, ll_keywords=ll, only_need_context=True)
        res = await rag.aquery_llm(cli_question, param=param)
        print("\n================ RETRIEVED GRAPH CONTEXT ================")
        print(res.get("llm_response", {}).get("content", ""))
        print("=========================================================\n")
        return

    if cli_question:
        print(f"\n🤖 Asking GraphRAG ({LLM_MODEL} + Neo4j): {cli_question}\n")
        print("================ GRAPHRAG GROUNDED ANSWER ================")
        await ask_graphrag(cli_question)
        print("==========================================================\n")
        return

    # Check unindexed count to inform user in interactive session
    try:
        unindexed_res = supabase.table("scraped_pages").select("id", count="exact").eq("graph_synced", False).execute()
        unindexed_count = unindexed_res.count or len(unindexed_res.data or [])
    except Exception:
        unindexed_count = 0

    # Interactive prompt loop in cmd/terminal
    print("\n" + "=" * 60)
    print(f"💬 GraphRAG Console (Local Ollama: {LLM_MODEL} + Neo4j Aura)")
    print("Modes:")
    print("  • Type any question for fast streaming GraphRAG answer")
    print("  • Type '/search <term>' or '/fts <term>' for instant (<50ms) Neo4j lookup")
    print("  • Type '/context <term>' for raw retrieved subgraph context")
    if unindexed_count > 0:
        print(f"  • Type '/sync' to index remaining {unindexed_count} Supabase page(s)")
    print("  • Type 'exit' or 'quit' to end")
    print("=" * 60)

    while True:
        try:
            user_question = await asyncio.to_thread(input, "\n❓ Ask a question: ")
            user_question = user_question.strip()

            if not user_question:
                continue

            if user_question.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break

            if user_question.lower() in ("/sync", "sync"):
                await sync_supabase_to_graph()
                continue

            # Instant FTS lookup command
            if user_question.lower().startswith(("/search ", "/fts ", "/find ")):
                query_term = user_question.split(" ", 1)[1].strip()
                print(f"\n⚡ [Instant Neo4j FTS (<50ms)]: Searching '{query_term}'...")
                results = fast_graph_fts_search(query_term)
                print("\n" + format_fts_results(results))
                continue

            # Instant context retrieval command
            if user_question.lower().startswith("/context "):
                query_term = user_question.split(" ", 1)[1].strip()
                hl, ll = extract_keywords_fast(query_term)
                param = QueryParam(mode="local", hl_keywords=hl, ll_keywords=ll, only_need_context=True)
                res = await rag.aquery_llm(query_term, param=param)
                print("\n================ RETRIEVED GRAPH CONTEXT ================")
                print(res.get("llm_response", {}).get("content", ""))
                print("=========================================================")
                continue

            # Fast Auto-detect: if user just enters a project code like 24HH0043
            if re.fullmatch(r'\d{2}[A-Za-z]{2}\d{4}', user_question, re.IGNORECASE):
                print(f"\n⚡ Detected project ID '{user_question}' - Running instant Neo4j lookup (<50ms)...")
                results = fast_graph_fts_search(user_question)
                print("\n" + format_fts_results(results))
                continue

            print("\n🔍 Retrieving from Neo4j & streaming grounded answer...")
            print("\n================ GRAPHRAG GROUNDED ANSWER ================")
            await ask_graphrag(user_question)
            print("==========================================================")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting interactive session.")
            break
        except Exception as e:
            print(f"\n❌ Error answering question: {e}")

if __name__ == "__main__":
    asyncio.run(main())