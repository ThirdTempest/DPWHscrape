"""
DPWH Cebu Infrastructure Pipeline Package
==========================================
Includes:
- scrape_cebu_dpwh: Scraper for BetterGov / DPWH infrastructure projects in Cebu (2022-present)
- sync_to_neo4j: Batch ingestion into Neo4j Aura knowledge graph
- query_graph: Sub-second graph querying, instant insights, and grounded streaming LLM QA
"""

from .sync_to_neo4j import sync_projects_to_neo4j
from .scrape_cebu_dpwh import main as run_scraper
from .query_graph import (
    run_console,
    ask_llm_stream,
    search_project_by_id,
    search_graph_fts,
    get_top_contractors,
    get_top_projects,
    get_dataset_summary,
    print_project_card,
    print_matched_projects_list,
    print_instant_insights,
    print_fts_results,
)

__all__ = [
    "run_scraper",
    "sync_projects_to_neo4j",
    "run_console",
    "ask_llm_stream",
    "search_project_by_id",
    "search_graph_fts",
    "get_top_contractors",
    "get_top_projects",
    "get_dataset_summary",
    "print_project_card",
    "print_matched_projects_list",
    "print_instant_insights",
    "print_fts_results",
]
