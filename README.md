# DPWH Cebu Infrastructure Knowledge Graph & QA System

An end-to-end GraphRAG pipeline for scraping, ingesting, analyzing, and querying **4,370+ DPWH civil works projects** across Cebu Province (2022 – Present).

Built with **Neo4j Aura (Knowledge Graph)**, **Supabase (Document Lake)**, **Apify (Web Scraper)**, and **Local Ollama (`qwen2.5:1.5b`)** for $0-cost, private, sub-second grounded AI question answering.

---

## 📁 Repository Structure

```text
Apify/
├── main.py                     # ⚡ Master CLI runner (executes all tasks)
├── pipeline/                   # 📦 Core modular pipeline package
│   ├── __init__.py             # Package exports
│   ├── scrape_cebu_dpwh.py     # BetterGov scraper -> Apify & Supabase
│   ├── sync_to_neo4j.py        # Paginated batch ingestion -> Neo4j Aura
│   └── query_graph.py          # Sub-second Cypher queries, Instant Insights & Ollama LLM
├── .env.example                # Template for required environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Installation & usage guide
```

---

## 🛠️ Prerequisites

Before installing, ensure you have the following installed on your machine:

1. **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14)
2. **Git**
3. **[Ollama](https://ollama.com/)** (Required for local LLM question answering)
4. Free Accounts for:
   * **[Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/)** (Free cloud graph database instance)
   * **[Supabase](https://supabase.com/)** (Free PostgreSQL/storage instance)
   * **[Apify](https://apify.com/)** (Optional: only needed if running the scraper)

---

## 🚀 Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/ThirdTempest/DPWHscrape.git
cd DPWHscrape
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables

Create your `.env` file by copying the template:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```dotenv
# --- Apify & Supabase (Scraper & Document Storage) ---
APIFY_TOKEN="your_apify_api_token"
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_SERVICE_KEY="your_supabase_service_role_key"

# --- Neo4j Graph Database (AuraDB Free or Local Desktop) ---
NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_neo4j_password"

# --- Local Ollama Configuration ---
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL="qwen2.5:1.5b"
LLM_API_KEY="ollama"
```

---

### Step 5: (Optional) Set Up Supabase Table
If you are setting up a brand-new Supabase project and intend to run the scraper, execute this SQL query in your **Supabase SQL Editor**:

```sql
CREATE TABLE IF NOT EXISTS scraped_pages (
    id BIGSERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    markdown TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    content_hash TEXT,
    graph_synced BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scraped_pages_url ON scraped_pages(url);
CREATE INDEX IF NOT EXISTS idx_scraped_pages_synced ON scraped_pages(graph_synced);
```

---

### Step 6: Install & Start Local Ollama Model
In a separate terminal window, pull the lightweight, CPU-optimized model:

```bash
ollama pull qwen2.5:1.5b
```

Verify Ollama is running:
```bash
ollama list
```

---

## 💻 How to Run (Master CLI)

Everything is executed through [`main.py`](file:///C:/Users/User/VSCode_Programming/Apify/main.py):

### 1. Interactive Menu & QA Console
```powershell
python main.py
```
Press **Enter** to open the interactive QA console, or select options to sync, scrape, or view analytics.

---

### 2. Instant Lookups & Natural Language Questions
```powershell
# Instant single project card lookup (<100ms)
python main.py 25H00064
python main.py 24HH0043

# Natural language grounded query (streams summary in ~2.5s)
python main.py "any project in bogo?"
python main.py "is there any project in lapulapu?"

# Sub-second instant search mode (skips LLM wait on search lists)
python main.py --instant
```

---

### 3. Analytics & Graph Insights
```powershell
# View Top 10 Contractors by awarded budget
python main.py --top-contractors

# View Top 10 Largest Infrastructure Projects
python main.py --top-budgets

# View dataset totals
python main.py --summary
```

---

### 4. Data Scraping & Neo4j Sync
```powershell
# Sync all projects from Supabase into Neo4j Aura
python main.py --sync

# Scrape 4,370 projects from BetterGov into Apify & Supabase
python main.py --scrape

# Run full end-to-end pipeline: Scrape -> Sync -> Open Console
python main.py --all
```

---

## 🔧 Performance & Hardware Optimizations

* **Hardware**: Runs 100% on CPU (e.g. AMD Ryzen or Intel) without requiring an expensive NVIDIA GPU.
* **Low Latency Cypher**: Neo4j full-text queries apply `WITH node, score ORDER BY score DESC LIMIT $limit` before relationship expansion, dropping query latency from **607ms to 94ms**.
* **Instant Graph Insights**: Sub-millisecond aggregation calculates **Total Projects**, **Total Budget**, **Primary Sectors**, and **Key Builders** in **<1ms** directly from graph facts.
* **CPU Thread & Context Tuning**: Local Ollama runs with `num_thread: 6` and `num_ctx: 512`, cutting Time-To-First-Token from **6.7s down to 1.5s**.

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `'ollama' is not recognized` | Ensure Ollama is installed from [ollama.com](https://ollama.com) and added to your system `PATH` (on Windows: `C:\Users\<User>\AppData\Local\Programs\Ollama`). |
| `UnicodeEncodeError: 'charmap' codec...` | Run with Python's UTF-8 flag: `python -X utf8 main.py`. |
| `Missing required environment variables` | Verify your `.env` file exists in the project root and is populated. |
| `Neo4j connection error` | Check your Aura instance status in the [Neo4j Aura Console](https://console.neo4j.io/). Free instances automatically pause after 3 days of inactivity; simply click **Resume**. |
