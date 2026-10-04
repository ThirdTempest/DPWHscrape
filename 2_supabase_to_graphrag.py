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
from google import genai
from google.genai import types, errors
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Suppress internal library warning spam (e.g. AFC warnings, rerank notifications)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("lightrag").setLevel(logging.WARNING)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables in .env (GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 1. LLM Model Pool with automatic fallback
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]
MODELS_TO_TRY = list(dict.fromkeys(FALLBACK_MODELS))
current_model_idx = 0

# 2. Embedding Model Pool with automatic fallback
PRIMARY_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
FALLBACK_EMBEDDING_MODELS = [
    PRIMARY_EMBEDDING_MODEL,
    "gemini-embedding-2",
]
EMBEDDING_MODELS = list(dict.fromkeys(FALLBACK_EMBEDDING_MODELS))
current_embed_model_idx = 0

EMBEDDING_DIM = 768
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
    print("🧹 Cleaning out incomplete/failed graph cache and resetting sync status...")
    shutil.rmtree(WORKING_DIR, ignore_errors=True)
    supabase.table("scraped_pages").update({"graph_synced": False}).neq("url", "").execute()

os.makedirs(WORKING_DIR, exist_ok=True)

def is_daily_quota_error(err_msg: str) -> bool:
    """Detect if error is a daily quota exhaustion rather than a temporary per-minute burst."""
    # 1. If explicit retry delay is present, anything <= 60s is RPM (burst), anything > 60s is daily/hours
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) > 60.0
    match = re.search(r"retryDelay['\"]: ['\"]?(\d+)s?", err_msg)
    if match:
        return float(match.group(1)) > 60.0

    # 2. Check for explicit PerDay or daily keywords
    return any(k.lower() in err_msg.lower() for k in (
        "perday",
        "requests per day",
        "generaterequestsperday",
        "embedcontentrequestsperday",
        "retry in 1h", "retry in 2h", "retry in 3h", "retry in 4h", "retry in 5h",
        "retry in 6h", "retry in 7h", "retry in 8h", "retry in 9h", "retry in 10h", "retry in 24h"
    ))

def parse_retry_delay(err_msg: str, default: float = 5.0, max_delay: float = 20.0) -> float:
    """Extract retry delay seconds from Gemini 429 response, capped to prevent long freezes."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, max_delay)
    match = re.search(r"retryDelay['\"]: ['\"]?(\d+)s?", err_msg)
    if match:
        return min(float(match.group(1)) + 1.0, max_delay)
    return min(default, max_delay)

# 1. Optimized Gemini LLM Wrapper with auto-fallback and no-freeze error handling
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    global current_model_idx
    max_retries = 4

    while current_model_idx < len(MODELS_TO_TRY):
        model_name = MODELS_TO_TRY[current_model_idx]
        for attempt in range(max_retries):
            try:
                # Small pacing to stay smoothly under free-tier RPM
                await asyncio.sleep(0.5)
                return await gemini_model_complete(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=GEMINI_API_KEY,
                    model_name=model_name,
                    **kwargs,
                )
            except (errors.ServerError, errors.ClientError) as e:
                err_str = str(e)

                # Daily quota exhausted -> Immediately move to next model in pool without sleeping for hours!
                if is_daily_quota_error(err_str):
                    if current_model_idx < len(MODELS_TO_TRY) - 1:
                        current_model_idx += 1
                        next_model = MODELS_TO_TRY[current_model_idx]
                        print(f"🔄 Daily LLM quota reached for {model_name}. Switching to: {next_model}")
                        break
                    else:
                        print("\n❌ All Gemini LLM models in the fallback pool have reached their free daily limit.")
                        print("👉 Please wait for the daily quota reset or use a paid Gemini API key.")
                        raise RuntimeError("All available Gemini models reached daily free quota.") from e

                # Transient 503 unavailable or temporary 429 per-minute rate limit
                if attempt < max_retries - 1 and any(c in err_str for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    delay = parse_retry_delay(err_str, default=(attempt + 1) * 3.0, max_delay=20.0)
                    print(f"⚠️ Rate limited with {model_name}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                # If attempts failed, try next model if available
                if current_model_idx < len(MODELS_TO_TRY) - 1:
                    current_model_idx += 1
                    next_model = MODELS_TO_TRY[current_model_idx]
                    print(f"🔄 Error with {model_name}. Trying next model: {next_model}")
                    break
                raise

# 2. Optimized Gemini Batch Embedding Wrapper with auto-fallback and batching
async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    global current_embed_model_idx
    batch_size = 25
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        max_retries = 5
        batch_embedded = False

        while current_embed_model_idx < len(EMBEDDING_MODELS) and not batch_embedded:
            model_name = EMBEDDING_MODELS[current_embed_model_idx]
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(0.5)
                    res = await gemini_client.aio.models.embed_content(
                        model=model_name,
                        contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in batch],
                        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
                    )
                    for e in res.embeddings:
                        v = np.array(e.values, dtype=np.float32)
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            v /= norm
                        all_vectors.append(v)
                    batch_embedded = True
                    break
                except (errors.ServerError, errors.ClientError) as e:
                    err_str = str(e)

                    # Daily quota error -> immediately switch to next embedding model
                    if is_daily_quota_error(err_str):
                        if current_embed_model_idx < len(EMBEDDING_MODELS) - 1:
                            current_embed_model_idx += 1
                            next_model = EMBEDDING_MODELS[current_embed_model_idx]
                            print(f"🔄 Daily embedding quota reached for {model_name}. Switching to: {next_model}")
                            break
                        else:
                            print("\n❌ All Gemini embedding models reached daily free quota.")
                            raise RuntimeError("All available Gemini embedding models reached daily free quota.") from e

                    # Transient RPM rate-limit or 503
                    if attempt < max_retries - 1 and any(c in err_str for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                        delay = parse_retry_delay(err_str, default=(attempt + 1) * 5.0, max_delay=60.0)
                        print(f"⚠️ Embedding API rate-limited with {model_name}. Waiting {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue

                    # If retries failed on transient error, try next model
                    if current_embed_model_idx < len(EMBEDDING_MODELS) - 1:
                        current_embed_model_idx += 1
                        next_model = EMBEDDING_MODELS[current_embed_model_idx]
                        print(f"🔄 Error with embedding model {model_name}. Trying next: {next_model}")
                        break
                    raise

    return np.array(all_vectors, dtype=np.float32)

# 3. Initialize LightRAG with single-worker concurrency, 25-item batching, and generous timeouts
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_name=MODELS_TO_TRY[0],
    llm_model_func=llm_model_func,
    llm_model_max_async=1,
    embedding_func_max_async=1,
    embedding_batch_num=25,
    default_embedding_timeout=300,
    default_llm_timeout=300,
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

async def sync_supabase_to_graph():
    """Sync unindexed rows from Supabase into LightRAG Knowledge Graph."""
    response = supabase.table("scraped_pages").select("*").eq("graph_synced", False).execute()
    rows = response.data

    if not rows:
        print("✅ Knowledge graph is up-to-date (all Supabase pages indexed).")
        return

    print(f"🔍 Found {len(rows)} unindexed page(s) in Supabase. Syncing to Knowledge Graph...")
    for idx, row in enumerate(rows, 1):
        print(f"\n⚙️ [{idx}/{len(rows)}] Indexing: {row.get('title') or row['url']}...")
        document_with_citation = (
            f"SOURCE_URL: {row['url']}\n"
            f"PAGE_TITLE: {row.get('title') or 'Untitled'}\n"
            f"CONTENT:\n{row['markdown']}"
        )

        try:
            track_id = await rag.ainsert(document_with_citation)
            doc_statuses = await rag.aget_docs_by_track_id(track_id)

            failed_docs = [
                (doc_id, status)
                for doc_id, status in doc_statuses.items()
                if str(getattr(status, "status", "")).lower() not in ("docstatus.processed", "processed")
            ]

            if not failed_docs:
                supabase.table("scraped_pages").update({"graph_synced": True}).eq("id", row["id"]).execute()
                print(f"✅ Indexed & marked synced: {row['url']}")
            else:
                for doc_id, status in failed_docs:
                    print(f"❌ Failed to index {doc_id} ({row['url']}): {getattr(status, 'error_msg', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error processing {row['url']}: {e}")

async def main():
    await sanitize_and_sync_cache()
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Check command line arguments
    sync_requested = "--sync" in sys.argv
    cli_question = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")]).strip()

    if sync_requested:
        await sync_supabase_to_graph()
        if not cli_question:
            return

    if cli_question:
        # One-shot query mode
        print(f"\n🤖 Asking GraphRAG: {cli_question}\n")
        answer = await rag.aquery(cli_question, param=QueryParam(mode="hybrid", enable_rerank=False))
        print("\n================ GRAPHRAG GROUNDED ANSWER ================")
        print(answer)
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
    print("💬 GraphRAG Interactive Console")
    print("Ask any question to test your knowledge graph.")
    if unindexed_count > 0:
        print(f"ℹ️  {unindexed_count} unindexed page(s) in Supabase. Type '/sync' to index them.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
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

            print("\n🔍 Searching knowledge graph & generating grounded answer...")
            answer = await rag.aquery(user_question, param=QueryParam(mode="hybrid", enable_rerank=False))
            print("\n================ GRAPHRAG GROUNDED ANSWER ================")
            print(answer)
            print("=" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting interactive session.")
            break
        except Exception as e:
            print(f"\n❌ Error answering question: {e}")

if __name__ == "__main__":
    asyncio.run(main())