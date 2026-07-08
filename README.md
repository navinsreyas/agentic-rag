# Agentic RAG with Knowledge Graph and Semantic PageIndex

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?logo=neo4j&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-412991?logo=openai&logoColor=white)
![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-0.3.2-E92063?logo=pydantic&logoColor=white)

---

## Overview

A production-grade document intelligence system that answers questions over a small corpus of AI-safety and big-tech AI-strategy documents using three complementary retrieval strategies: hybrid vector search, a temporal knowledge graph, and full-page vector search. The agent — built on Pydantic AI with Groq Llama-3.3-70b as the primary model — routes each incoming question to the appropriate tool based on the question type, then synthesises a cited response from the retrieved context. The backend uses PostgreSQL with pgvector for chunk and page-level embeddings, Neo4j with Graphiti for temporal entity-relationship extraction, and FastAPI for streaming and non-streaming HTTP endpoints. Ingestion is split into two decoupled pipelines: a fast embedding-only run that completes in minutes, and an overnight graph-building run that safely tracks progress across restarts.

---

## Project Status

### What's Complete ✅
- Vector search + hybrid search — working
- Full-page vector search — working, 158 page embeddings
- Knowledge graph — populated (294 episodes, 1,241 entities, 1,610 relationship facts in Neo4j)
- FastAPI server + CLI — working
- Groq as primary LLM — configured and working
- OpenAI fallback — implemented (auto-retry on tool-call / rate-limit errors)
- All 9 tools registered and named correctly in agent + system prompt

### What's Pending ⚠️
- **Graph coverage is partial** — the graph was built from a subset of chunks (`graph_progress.json` tracks resumable progress). Graph tools (`graph_search`, `get_entity_relationships`, `get_entity_timeline`) are wired, functional, and return data, but coverage does not span every chunk.
- **RSP document** — Anthropic's Responsible Scaling Policy PDF has a null-byte encoding issue and is **not** among the 5 ingested documents. Pending a fix with `pdfplumber` or `pymupdf`.
- **o1-system-card.pdf** — not ingested; it shares the title "OpenAI o1 System Card" with the already-ingested `OpenAI-2024.pdf`, and title-based dedup skips it.
- **Evaluation** — No automated metrics. All testing has been manual via the CLI.
- **Frontend** — The streaming API is production-ready but there is no React/Next.js UI yet. The CLI is the primary interface.

---

## Architecture

```
                                 User Query
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │        FastAPI  (api.py)      │
                     │  /chat   /chat/stream         │
                     │  /search/{vector,graph,hybrid}│
                     └───────────────┬──────────────-┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │       Pydantic AI Agent       │
                     │  Primary : Groq Llama-3.3-70b │
                     │  Fallback: OpenAI gpt-4o-mini │
                     │ 9 tools · 1/response, ≤2/turn │
                     └──┬───────┬────────┬─────────┬─┘
                        │       │        │         │
      ┌─────────────────▼┐ ┌────▼───────┐ ┌▼─────────────────┐ ┌▼───────────────┐
      │ vector_search    │ │graph_search│ │page_vector_search│ │pageindex_search│
      │ hybrid_search    │ │get_entity_ │ │ (full PDF pages) │ │ (VectifyAI     │
      │                  │ │relationships│ │                  │ │  cloud tree)   │
      │                  │ │get_entity_ │ │                  │ │                │
      │                  │ │timeline    │ │                  │ │                │
      └────────┬─────────┘ └─────┬──────┘ └────────┬─────────┘ └───────┬────────┘
               │                 │                 │                   │
               ▼                 ▼                 ▼                   ▼
      ┌────────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────────────┐
      │   PostgreSQL   │ │    Neo4j     │ │  PostgreSQL    │ │ api.pageindex  │
      │     chunks     │ │  + Graphiti  │ │ page_embeddings│ │    .ai         │
      │    pgvector    │ │ Temporal KG  │ │   pgvector     │ │  (cloud API)   │
      └────────────────┘ └──────────────┘ └────────────────┘ └────────────────┘

      Document tools (9th/8th): get_document, list_documents → PostgreSQL documents/chunks
                                     │
                                     ▼
                            Synthesised Response
                          (with document citations)
```

---

## The Retrieval Tools

The agent registers **9 tools** in total: two vector tools, three graph tools, two page-level tools, and two document tools.

**Tool-calling policy:** the system prompt enforces exactly one tool call per model response. Because a question can need a follow-up retrieval, a hard per-turn cap (`_MAX_TOOL_CALLS_PER_TURN = 2` in `agent.py`) permits at most one additional call in a later response and then forces the agent to answer — this prevents the tool-call loops that Llama occasionally falls into. So: **one tool per response, at most two per question.**

### 1. Vector / Hybrid Search — `vector_search`, `hybrid_search`

Documents are split into character-based chunks (the ingestion default is ~1000 characters with 200-character overlap; the 552 stored chunks average ~730 characters ≈ 180 tokens each), embedded with `text-embedding-3-small`, and stored in a `chunks` table with a `pgvector` index. When a user asks a broad conceptual question — "What is Constitutional AI?", "What challenges does the paper identify?" — the agent embeds the query and retrieves the top-*k* chunks by cosine similarity. `hybrid_search` adds a BM25-style keyword component, which helps when the query contains specific terminology that may not be semantically unique. This is the highest-recall path and the first tool used for most general questions.

### 2. Knowledge Graph — `graph_search`, `get_entity_relationships`, `get_entity_timeline`

Each document chunk is also passed through Graphiti, which extracts named entities and their temporal relationships and stores them as episodes in Neo4j. This creates a structured, queryable layer on top of the unstructured text. `graph_search` performs semantic search directly over the graph's fact nodes; `get_entity_relationships` runs a focused semantic search for the facts involving a named entity; `get_entity_timeline` returns the chronological sequence of facts about an entity (newest first), with optional inclusive `start_date`/`end_date` filtering on each fact's `valid_at`. These tools excel at questions like "How does chain-of-thought reasoning relate to interpretability?" or "How has OpenAI's approach to safety evolved?" — questions that require understanding connections between concepts, not just finding similar text.

### 3. Full-Page Vector Search — `page_vector_search`

Standard chunk retrieval suffers from context fragmentation: a methodology that spans three paragraphs gets split across multiple chunks, none of which individually scores highly enough to be retrieved. Full-page vector search solves this by embedding and storing complete PDF pages as atomic units in a `page_embeddings` table. When the agent calls `page_vector_search`, it retrieves the 3 pages with the highest semantic (embedding) similarity to the query and returns them in full — preserving numbered steps, tables, and multi-paragraph arguments that chunks would sever. This is the right tool whenever the question demands precise wording, exact statistics, or a complete section rather than a synthesised excerpt.

> **Naming note:** this tool is *full-page vector search* — it ranks whole pages by embedding similarity. It is **not** the vectorless, tree-reasoning "PageIndex" technique. That is a separate tool, `pageindex_search` (below).

### 4. PageIndex cloud reasoning — `pageindex_search`

`pageindex_search` calls VectifyAI's PageIndex cloud API (`api.pageindex.ai`). Instead of embedding similarity, the service builds a hierarchical section tree of each uploaded PDF and uses server-side LLM reasoning to navigate that tree top-down to the specific section that answers the query. It is the right tool for precise section lookups in structured documents ("what does the policy say about X", "according to section Y…"). Uploaded documents are tracked in `pageindex_trees/index.json`; requires `PAGEINDEX_API_KEY` and the `pageindex` package.

### Document tools — `get_document`, `list_documents`

`list_documents` returns every ingested document with its title, source, and chunk count; `get_document` fetches one document's full content plus all its chunks by UUID. These support "what sources are available?" and "retrieve the full X document" style requests.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | Pydantic AI 0.3.2 |
| LLM (primary) | Groq — Llama-3.3-70b-versatile |
| LLM (fallback) | OpenAI — gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Vector database | PostgreSQL 15 + pgvector |
| Graph database | Neo4j 5.x |
| Graph extraction | Graphiti 0.12 (temporal entity-relationship) |
| API server | FastAPI 0.115 + uvicorn (SSE streaming) |
| Cloud database | Neon (serverless PostgreSQL) |
| Schema management | Manual SQL (`sql/schema.sql`) |
| PDF parsing | PyPDF2 |
| Semantic chunking | Custom `SemanticChunker` (section-aware) |

---

## Project Stats

All counts below are read directly from the live databases (Postgres + Neo4j).

| Metric | Value |
|---|---|
| Documents ingested | 5 (3 PDFs, 2 markdown) |
| Pages indexed | 158 full-page embeddings |
| Chunks stored | 552 at 1536-dimension embeddings |
| Source material | Anthropic safety-research info sheet, OpenAI o1 system card, a redacted risk report (PDFs); Apple and Google AI-strategy notes (markdown) |
| Embedding dimensions | 1536 (text-embedding-3-small) |
| Knowledge graph | 294 episodes, 1,241 entities, 1,610 relationship facts (Neo4j) |

> **Note on the repo's document folders:** `documents/` holds the 7 source files, of which 5 are currently ingested (see caveats above). `big_tech_docs/` contains 19 additional big-tech funding/business markdown files that are **not** ingested by the default pipeline — `scripts/ingest_fast.py` only reads the `--documents` folder (default `documents/`). To include them, run `python scripts/ingest_fast.py --documents big_tech_docs/` as a second pass.

---

## Setup

### 1. Clone and create environment

```bash
git clone <repository-url>
cd agentic-rag-knowledge-graph

conda create -n rag-kg python=3.11
conda activate rag-kg
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in the following required keys:

```bash
# PostgreSQL (Neon or self-hosted)
DATABASE_URL=

# Neo4j
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=

# Primary LLM (Groq recommended)
LLM_PROVIDER=
LLM_BASE_URL=
LLM_API_KEY=
LLM_CHOICE=

# Embeddings (OpenAI)
EMBEDDING_PROVIDER=
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=

# Fallback LLM — optional, but recommended when using Groq
# Set to openai + gpt-4o-mini to auto-retry on tool-call failures
FALLBACK_LLM_PROVIDER=
FALLBACK_LLM_API_KEY=
FALLBACK_LLM_CHOICE=
FALLBACK_LLM_BASE_URL=
```

### 3. Initialise the database schema

Run `sql/schema.sql` against your PostgreSQL instance. This creates the `documents`, `chunks`, `page_embeddings`, `sessions`, and `messages` tables (the `messages` table is required by `agent/api.py` for session history) with the correct pgvector indexes and the `match_pages()` SQL function.

> **Note:** If your embedding model is not `text-embedding-3-small`, update all `vector(1536)` occurrences in `schema.sql` to match your model's output size before running (currently lines 35, 58, and 71). Line numbers drift as the file changes — `grep -n 'vector(1536)' sql/schema.sql` to find them all.

### 4. Ingest documents (fast pipeline — embeddings only)

Place your PDFs and markdown files in a `documents/` folder, then run:

```bash
# Dry run: shows cost estimate before any API calls
python scripts/ingest_fast.py --dry-run

# Full run (prompts for confirmation)
python scripts/ingest_fast.py --documents documents/

# Backfill page embeddings for documents already in the database
python scripts/ingest_fast.py --force-pages
```

This completes in 5–10 minutes and makes vector search and full-page vector search immediately usable.

### 5. Ingest into the knowledge graph (optional, overnight)

```bash
# Process up to 50 chunks at a time, safe to Ctrl-C and restart
python scripts/ingest_graph.py --limit 50
```

Progress is tracked in `graph_progress.json`. Re-running the script skips already-processed chunks.

### 6. (Optional) Index PDFs into the PageIndex cloud

The `pageindex_search` tool queries VectifyAI's PageIndex cloud API and only works if your PDFs have been uploaded and their `doc_id`s recorded in `pageindex_trees/index.json`. Run this after step 4 if you want that tool:

```bash
# Requires PAGEINDEX_API_KEY in .env (get one at https://pageindex.ai)
python ingestion/ingest_pageindex.py
```

If you skip this, `pageindex_search` returns a "no documents indexed" message and the agent falls back to `page_vector_search`. The other eight tools work without it.

### 7. Start the API server

```bash
python -m agent.api
# Server available at http://localhost:8058
# Interactive docs at http://localhost:8058/docs
```

### 8. Use the CLI

```bash
# In a second terminal
python cli.py

# Connect to a custom port
python cli.py --port 8058
```

---

## Example Queries

| Query | Tool triggered | Sample answer |
|---|---|---|
| "What is Constitutional AI?" | `vector_search` | "Constitutional AI is a training methodology developed by Anthropic in which an AI model critiques and revises its own outputs against a set of principles..." |
| "What does the RSP say about ASL-3 safety requirements?" | `page_vector_search` | Returns the full RSP page defining ASL-3 thresholds — including the exact capability criteria and required mitigations — without truncation |
| "How is chain-of-thought reasoning connected to interpretability?" | `graph_search` | "The knowledge graph links chain-of-thought reasoning to mechanistic interpretability via shared nodes around transparency and model behaviour..." |
| "What connects RLHF to AI alignment?" | `get_entity_relationships` | "RLHF connects to Constitutional AI (refinement), InstructGPT (application), and human feedback mechanisms (dependency)..." |
| "How has OpenAI's approach to safety evolved over time?" | `get_entity_timeline` | "2022: InstructGPT introduces RLHF as primary alignment technique. 2023: GPT-4 system card describes multi-layered red-teaming. 2024: Preparedness Framework defines tiered risk thresholds..." |
| "Summarise the methodology for evaluating dangerous capabilities" | `page_vector_search` | Returns the complete evaluation pipeline pages, preserving the numbered steps and decision criteria intact |

---

## Design Decisions

### Why full-page vector search over pure chunking

The core limitation of chunk-based retrieval is context fragmentation. A semantic chunker splits documents at token boundaries — which means a 600-token methodology section spanning four paragraphs will be divided across three chunks, none of which individually contains a complete argument. When the agent retrieves the top-5 chunks for a query like "explain the full evaluation process", it may receive disjointed fragments that require significant reconstruction before they become usable.

Full-page vector search treats each PDF page as an atomic retrieval unit. Pages are embedded whole and stored in a separate `page_embeddings` table. When the agent calls `page_vector_search`, it retrieves 1–3 complete pages rather than assembled fragments — preserving tables, numbered steps, and cross-sentence arguments as the author intended. The trade-off is a larger per-call context, but this is acceptable because the agent only routes to this tool when the question explicitly requires full-context fidelity, not general semantic similarity.

### Why separate ingestion from graph building

Knowledge graph ingestion via Graphiti makes **several LLM calls per chunk — roughly 4–5** (entity extraction, entity deduplication, fact/edge extraction, edge deduplication, and temporal resolution), not one. With 552 chunks in the current corpus that is on the order of 2,000–2,800 sequential LLM calls. At Groq's free-tier rate limits, this runs for many hours.

If vector search and graph building were coupled in a single pipeline, a rate-limit error three hours in would force a full restart. The decoupled design solves this: `scripts/ingest_fast.py` runs the embedding-only pipeline to completion in 5–10 minutes, making vector search and the PageIndex immediately available. `scripts/ingest_graph.py` runs separately and tracks each processed chunk ID in `graph_progress.json`, so the script can be interrupted at any point and resumed from where it left off — with `--limit N` to control how many chunks are processed per session.

### Why Groq for LLM with OpenAI fallback

Groq's inference hardware runs Llama-3.3-70b at effectively zero cost on the free tier, which is the right choice for a research project where query volume is low and the priority is minimising API spend while using a capable model. The practical trade-off is that Groq's Llama implementation occasionally produces malformed tool-call responses — specifically, the model sometimes attempts to call two tools simultaneously, which pydantic-ai's streaming parser receives as a concatenated tool name (e.g., `graph_search,{"query":...}`).

Rather than switching entirely to a paid provider, the system registers a fallback: if the primary model raises a tool-call validation error or a rate-limit error, the same query is automatically retried with OpenAI's gpt-4o-mini, which has robust function-calling support. This gives near-zero running cost for the 90%+ of queries that succeed cleanly on Groq, with reliable degradation to a cheap paid model for the remainder. The fallback is implemented in `agent/providers.py` via `is_fallback_error()` and `get_fallback_model()`, and applied at both the streaming and non-streaming call sites in `agent/api.py`.

---

## Known Limitations

1. **Knowledge graph coverage is partial.** The graph is populated (294 episodes, 1,241 entities, 1,610 relationship facts), but `ingest_graph.py` has not processed every chunk in `chunks`, so graph tools (`graph_search`, `get_entity_relationships`, `get_entity_timeline`) may return sparse results for topics whose chunks were not yet graph-enriched.

2. **RSP document has a UTF-8 encoding issue.** Anthropic's Responsible Scaling Policy PDF contains characters that PyPDF2 cannot decode cleanly on Windows (cp1252 environment). Some pages may have garbled text in the extracted content. A fix using `pdfplumber` or `pymupdf` is pending.

3. **Groq free tier has a daily token limit.** The free tier caps total daily token throughput across all calls. Heavy use of the knowledge graph pipeline (`scripts/ingest_graph.py`) will exhaust the daily allowance in a single session. The `--limit` flag on `ingest_graph.py` is the recommended mitigation.

4. **No evaluation metrics.** There is currently no automated pipeline for measuring retrieval quality (precision@k, recall) or answer quality (faithfulness, relevance). All evaluation has been manual. See Future Work.

---

## Future Work

- **Ragas evaluation framework** — instrument the retrieval pipeline with [Ragas](https://docs.ragas.io/) to get quantitative metrics on context precision, recall, and answer faithfulness across the three retrieval paths.
- **Alembic migrations** — replace the manual `schema.sql` with Alembic-managed migrations to support incremental schema changes without dropping and recreating tables.
- **Additional source documents** — expand the corpus beyond Anthropic and OpenAI to include DeepMind, UK AI Safety Institute, and MIRI research, increasing graph density and retrieval coverage.
- **Streaming responses to frontend** — build a React or Next.js frontend that consumes the `/chat/stream` SSE endpoint and renders the `{"type": "info"}` fallback events and tool-usage metadata inline with the streamed response.

---

## Project Structure

```
agentic-rag-knowledge-graph/
├── agent/
│   ├── agent.py          # Pydantic AI agent — 9 registered tools
│   ├── api.py            # FastAPI app — streaming + non-streaming endpoints
│   ├── providers.py      # LLM provider abstraction (Groq / OpenAI / fallback)
│   ├── prompts.py        # System prompt + tool routing rules
│   ├── tools.py          # Tool implementations (vector, graph, hybrid)
│   ├── graph_utils.py    # Graphiti / Neo4j client wrapper
│   ├── db_utils.py       # asyncpg connection pool + query helpers
│   └── models.py         # Pydantic request / response models
├── ingestion/
│   ├── ingest.py         # Combined ingestion pipeline (library)
│   ├── chunker.py        # Section-aware SemanticChunker
│   ├── embedder.py       # Embedding generation with retry + caching
│   └── graph_builder.py  # Graphiti episode builder
├── scripts/
│   ├── ingest_fast.py    # Embeddings-only ingestion — 5–10 min, no LLM
│   └── ingest_graph.py   # Graph ingestion — restartable overnight run
├── page_index_semantic.py # PageIndex: ingest + query full-page embeddings
├── cli.py                # Interactive streaming CLI
├── sql/
│   └── schema.sql        # PostgreSQL schema + pgvector indexes
├── tests/                # pytest suite (agent + ingestion)
├── .env.example          # Environment variable template
└── requirements.txt
```

---

## Running Tests

```bash
pytest

# With coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Specific suites
pytest tests/agent/
pytest tests/ingestion/
```

---

## API Reference

Interactive Swagger docs are available at `http://localhost:8058/docs` once the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Connection status for PostgreSQL and Neo4j |
| `/chat` | POST | Non-streaming agent response |
| `/chat/stream` | POST | Server-Sent Events streaming response |
| `/search/vector` | POST | Direct vector search (bypasses agent) |
| `/search/graph` | POST | Direct graph search (bypasses agent) |
| `/search/hybrid` | POST | Direct hybrid search (bypasses agent) |
| `/documents` | GET | List all ingested documents |
| `/sessions/{id}` | GET | Retrieve session history |
