#!/usr/bin/env python3
"""
DPWH Cebu Infrastructure Master Runner
======================================
Unified entry point for scraping, syncing, analyzing, and querying
the DPWH Cebu Knowledge Graph (2022 - Present).

Usage:
  python main.py                            # Interactive menu / console
  python main.py 25H00064                   # Instant project card (<100ms)
  python main.py "any project in bogo?"     # Direct AI natural language query
  python main.py --instant                  # Sub-second search mode (no LLM wait)
  python main.py --sync                     # Sync Supabase data into Neo4j Aura
  python main.py --scrape                   # Scrape BetterGov to Apify & Supabase
  python main.py --all                      # Run full pipeline: Scrape -> Sync -> Console
  python main.py --top-contractors          # View top contractors by budget
  python main.py --summary                  # View total dataset statistics
"""

import sys
import os
import re
import asyncio
import argparse

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    run_console,
    ask_llm_stream,
    search_project_by_id,
    search_graph_fts,
    get_top_contractors,
    get_top_projects,
    get_dataset_summary,
    print_project_card,
    print_fts_results,
    sync_projects_to_neo4j,
    run_scraper,
)


def check_neo4j_credentials() -> bool:
    """Validate that Neo4j environment variables exist."""
    return bool(os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD"))


def check_scraper_credentials() -> bool:
    """Validate that Apify and Supabase environment variables exist."""
    return bool(os.getenv("APIFY_TOKEN") and os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def check_sync_credentials() -> bool:
    """Validate that Supabase and Neo4j environment variables exist."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY") and check_neo4j_credentials())


def show_neo4j_missing_error():
    print("\n" + "=" * 68)
    print("❌ CONFIGURATION ERROR: Missing Neo4j Database Credentials (.env)")
    print("=" * 68)
    print("The DPWH Cebu Knowledge Graph requires a Neo4j connection.")
    print("\n👉 Quick Setup Steps:")
    print("  1. Copy the example environment file:")
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


def show_scraper_missing_error():
    print("\n" + "=" * 68)
    print("❌ CONFIGURATION ERROR: Missing Scraper Credentials (.env)")
    print("=" * 68)
    print("Scraping DPWH projects requires Apify and Supabase credentials:")
    print("\n👉 Required in .env:")
    print('  APIFY_TOKEN="your_token"           (https://console.apify.com)')
    print('  SUPABASE_URL="https://xxx.supabase.co"')
    print('  SUPABASE_SERVICE_KEY="your_key"')
    print("\n💡 Tip: If you only want to query the existing knowledge graph,")
    print("   run: python main.py (no scraper keys needed!)")
    print("=" * 68 + "\n")


def show_sync_missing_error():
    print("\n" + "=" * 68)
    print("❌ CONFIGURATION ERROR: Missing Sync Credentials (.env)")
    print("=" * 68)
    print("Syncing Supabase projects to Neo4j requires both database connections:")
    print("\n👉 Required in .env:")
    print('  SUPABASE_URL="https://xxx.supabase.co"')
    print('  SUPABASE_SERVICE_KEY="your_key"')
    print('  NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"')
    print('  NEO4J_PASSWORD="your-password"')
    print("=" * 68 + "\n")



def print_banner():
    print("\n" + "=" * 68)
    print(" 🏗️  DPWH CEBU INFRASTRUCTURE KNOWLEDGE GRAPH & QA SYSTEM")
    print("     Unified Pipeline & Local Ollama Engine (2022 - Present)")
    print("=" * 68)


def show_menu():
    print_banner()
    print("Select an option:")
    print("  [1] 💬 Launch Interactive Query & AI Console (Default)")
    print("  [2] ⚡ Launch Instant Search Console (<100ms, skips LLM wait)")
    print("  [3] 🔄 Sync Supabase Projects into Neo4j Aura")
    print("  [4] 🌐 Scrape DPWH Cebu Projects via BetterGov & Apify")
    print("  [5] 🚀 Full End-to-End Pipeline (Scrape -> Sync -> Console)")
    print("  [6] 🏆 View Top 10 Contractors by Total Awarded Budget")
    print("  [7] 💰 View Top 10 Largest Infrastructure Projects")
    print("  [8] 📊 View Dataset & Knowledge Graph Statistics")
    print("  [0] ❌ Exit")
    print("=" * 68)


def run_full_pipeline():
    if not check_scraper_credentials():
        show_scraper_missing_error()
        return
    if not check_sync_credentials():
        show_sync_missing_error()
        return

    print("\n" + "=" * 68)
    print("🚀 RUNNING FULL END-TO-END PIPELINE")
    print("=" * 68)
    print("\n[Step 1/2] Scraping Cebu DPWH projects to Apify & Supabase...")
    sys.argv = ["scrape_cebu_dpwh.py", "--no-sync"]
    run_scraper()

    print("\n[Step 2/2] Ingesting all projects into Neo4j Aura Knowledge Graph...")
    sync_projects_to_neo4j(reset=False)

    print("\n🎉 Pipeline Complete! Launching Knowledge Graph Console...")
    run_console()


def handle_menu_choice(choice: str) -> bool:
    choice = choice.strip()

    if choice in ("1", "", "console"):
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        run_console()
        return True

    elif choice in ("2", "instant"):
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        print("\n⚡ Starting in Instant Mode (<100ms responses)...")
        # Preload interactive console with default mode = instant
        os.environ["DEFAULT_CONSOLE_MODE"] = "instant"
        run_console()
        return True

    elif choice in ("3", "sync"):
        if not check_sync_credentials():
            show_sync_missing_error()
            return False
        print("\n🔄 Running Neo4j Aura batch synchronization...")
        sync_projects_to_neo4j(reset=False)
        return True

    elif choice in ("4", "scrape"):
        if not check_scraper_credentials():
            show_scraper_missing_error()
            return False
        limit_str = input("Enter max projects to scrape (press Enter for ALL 4,370 projects): ").strip()
        args = ["scrape_cebu_dpwh.py"]
        if limit_str.isdigit():
            args.extend(["--limit", limit_str])
        sys.argv = args
        run_scraper()
        return True

    elif choice in ("5", "all", "full"):
        run_full_pipeline()
        return True

    elif choice in ("6", "contractors"):
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        rows = get_top_contractors(10)
        print("\n🏆 Top 10 Contractors by Total Awarded Budget:")
        print("-" * 68)
        for i, r in enumerate(rows, 1):
            formatted_budget = f"₱{r['total_budget']:,}"
            print(f"{i:2d}. {r['contractor']:<42} | {r['project_count']:2d} projects | {formatted_budget:>16}")
        print("-" * 68)
        return True

    elif choice in ("7", "budgets", "largest"):
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        rows = get_top_projects(10)
        print("\n💰 Top 10 Largest Infrastructure Projects:")
        print("-" * 68)
        for i, r in enumerate(rows, 1):
            formatted_budget = f"₱{r['budget_num']:,}"
            print(f"{i:2d}. [{r['id']}] {formatted_budget:>16} | Contractor: {r.get('contractor') or 'N/A'}")
            print(f"    Location: {r.get('location') or 'N/A'}")
        print("-" * 68)
        return True

    elif choice in ("8", "summary", "stats"):
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        stats = get_dataset_summary()
        tot_budget = f"₱{stats['total_budget']:,}" if stats['total_budget'] else "₱0"
        print("\n📊 Knowledge Graph Summary:")
        print(f"   • Total Projects Indexed:    {stats['total_projects']}")
        print(f"   • Total Unique Contractors:  {stats['total_contractors']}")
        print(f"   • Total Allocated Budget:    {tot_budget}")
        print("-" * 68)
        return True

    elif choice in ("0", "exit", "quit", "q"):
        print("👋 Goodbye!")
        return False

    else:
        print(f"⚠️ Unknown option '{choice}'. Defaulting to interactive console...")
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return False
        run_console()
        return True


def main():
    # If arguments are provided on command line, dispatch directly
    if len(sys.argv) > 1:
        first_arg = sys.argv[1].strip()

        # Flag handlers
        if first_arg in ("--help", "-h"):
            print(__doc__)
            return

        if first_arg in ("--menu", "-m"):
            show_menu()
            choice = input("\nEnter choice [1]: ").strip()
            handle_menu_choice(choice)
            return

        if first_arg in ("--sync", "sync"):
            if not check_sync_credentials():
                show_sync_missing_error()
                return
            reset_flag = "--reset" in sys.argv
            sync_projects_to_neo4j(reset=reset_flag)
            return

        if first_arg in ("--scrape", "scrape"):
            if not check_scraper_credentials():
                show_scraper_missing_error()
                return
            sys.argv = sys.argv[1:]
            run_scraper()
            return

        if first_arg in ("--all", "all", "full"):
            run_full_pipeline()
            return

        # Database queries below require Neo4j credentials
        if not check_neo4j_credentials():
            show_neo4j_missing_error()
            return

        if first_arg in ("--top-contractors", "-c"):
            rows = get_top_contractors(10)
            print("\n🏆 Top 10 Contractors by Total Awarded Budget:")
            print("-" * 68)
            for i, r in enumerate(rows, 1):
                formatted_budget = f"₱{r['total_budget']:,}"
                print(f"{i:2d}. {r['contractor']:<42} | {r['project_count']:2d} projects | {formatted_budget:>16}")
            print("-" * 68)
            return

        if first_arg in ("--top-budgets", "-b"):
            rows = get_top_projects(10)
            print("\n💰 Top 10 Largest Infrastructure Projects:")
            print("-" * 68)
            for i, r in enumerate(rows, 1):
                formatted_budget = f"₱{r['budget_num']:,}"
                print(f"{i:2d}. [{r['id']}] {formatted_budget:>16} | Contractor: {r.get('contractor') or 'N/A'}")
                print(f"    Location: {r.get('location') or 'N/A'}")
            print("-" * 68)
            return

        if first_arg in ("--summary", "-s"):
            stats = get_dataset_summary()
            tot_b = f"₱{stats['total_budget']:,}" if stats['total_budget'] else "₱0"
            print(f"Projects: {stats['total_projects']}, Contractors: {stats['total_contractors']}, Total Budget: {tot_b}")
            return

        if first_arg in ("--console", "console"):
            run_console()
            return

        if first_arg in ("--instant", "instant"):
            os.environ["DEFAULT_CONSOLE_MODE"] = "instant"
            run_console()
            return

        # Direct project code match (e.g. 24HH0043, 25H00064)
        if re.fullmatch(r'\d{2}[A-Za-z]{1,2}\d{4,5}', first_arg, re.IGNORECASE):
            proj = search_project_by_id(first_arg)
            if proj:
                print_project_card(proj)
            else:
                print(f"⚠️ Project {first_arg} not found in Neo4j.")
            return

        # Direct Natural Language query
        if first_arg in ("--ask", "-a"):
            question = " ".join(sys.argv[2:]).strip()
            asyncio.run(ask_llm_stream(question))
            return

        # Fallback: treat entire argument string as natural language question or search
        full_query = " ".join(sys.argv[1:]).strip()
        asyncio.run(ask_llm_stream(full_query))
        return

    # No arguments passed: automatically launch Option [1] (Interactive Console)
    if not check_neo4j_credentials():
        show_neo4j_missing_error()
        return

    print_banner()
    print("🚀 Auto-launching Interactive Query & AI Console (Default Option [1])...")
    print("💡 (To access batch scraping, sync, or dataset stats menu, run: python main.py --menu)\n")
    try:
        handle_menu_choice("1")
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exiting.")


if __name__ == "__main__":
    main()
