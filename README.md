# Agentic RAG with Knowledge Graph

A production-grade document intelligence system that answers questions over a mixed corpus of AI-safety research and big-tech strategy documents, using three complementary retrieval strategies routed by an autonomous agent.

**Live demo (interactive API docs):** https://agentic-rag-1065091982503.us-east1.run.app/docs

[![Lint](https://github.com/navinsreyas/agentic-rag/actions/workflows/lint.yml/badge.svg)](https://github.com/navinsreyas/agentic-rag/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![Neo4j](https://img.shields.io/badge/Neo4j-Graphiti-008CC1)

---

## Overview

A Pydantic AI agent routes each question to the appropriate retrieval tool, then synthesises a cited answer from the retrieved context. Three retrieval paradigms are available:

- **Hybrid vector search** — cosine similarity + BM25 keyword ranking over document chunks
- **Temporal knowledge graph** — entity-relationship reasoning via Neo4j + Graphiti
- **Full-page vector search** — whole PDF pages retrieved as atomic units, preserving tables and multi-paragraph arguments

The backend is FastAPI (streaming + non-streaming), with PostgreSQL/pgvector for embeddings and Neo4j for the temporal graph. It is deployed on Google Cloud Run against managed Neon Postgres and Neo4j Aura, with secrets in GCP Secret Manager.

## Architecture

```text
                          User Query
                              |
                              v
              +------------------------------+
              |        FastAPI (api.py)       |
              |  /chat   /chat/stream         |
              |  /search/{vector,graph,hybrid}|
              +---------------+--------------+
                              |
                              v
              +------------------------------+
              |       Pydantic AI Agent       |
              |  Primary : Groq Llama-3.3-70b |
              |  Fallback: OpenAI gpt-4o-mini |
              +--+--------+--------+--------+-+
                 |        |        |        |
                 v        v        v        v
            +--------+ +------+ +-------+ +----------+
            | vector | |graph | | page  | |pageindex |
            | hybrid | |tools | |vector | | (cloud)  |
            +---+----+ +--+---+ +---+---+ +----+-----+
                |         |         |          |
                v         v         v          v
            +--------+ +------+ +--------+ +----------+
            |Postgres| |Neo4j | |Postgres| |pageindex |
            |pgvector| |+Graph| | pages  | | .ai API  |
            +--------+ +------+ +--------+ +----------+
```
## Retrieval Tools

The agent registers 9 tools: two vector, three graph, two page-level, and two document tools. The system prompt enforces one tool call per response, with a hard cap of two per turn to prevent tool-call loops.

| Tool | Purpose |
|---|---|
| `vector_search`, `hybrid_search` | Semantic + keyword chunk retrieval (pgvector) |
| `graph_search`, `get_entity_relationships`, `get_entity_timeline` | Temporal entity-relationship reasoning (Neo4j + Graphiti) |
| `page_vector_search` | Whole-page retrieval, preserving full context |
| `pageindex_search` | VectifyAI PageIndex cloud API — vectorless, tree-reasoning section lookup |
| `get_document`, `list_documents` | Direct document access |

> **Naming note:** `page_vector_search` ranks whole pages by embedding similarity — it is *not* the vectorless "PageIndex" technique. That is a separate tool, `pageindex_search`, which calls VectifyAI's cloud API.

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | Pydantic AI 0.3.2 |
| LLM (primary / fallback) | Groq Llama-3.3-70b / OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Vector database | PostgreSQL + pgvector (Neon) |
| Graph database | Neo4j + Graphiti (Aura) |
| API | FastAPI 0.115 + uvicorn (SSE) |
| Deployment | Google Cloud Run + Secret Manager |
| Observability | Prometheus `/metrics`, Sentry |
| CI | Ruff + MyPy (GitHub Actions) |

## Project Stats

All counts read directly from the live databases.

| Metric | Value |
|---|---|
| Documents ingested | 5 (3 PDF, 2 markdown) |
| Chunks stored | 552 (1536-dim embeddings) |
| Page embeddings | 158 |
| Knowledge graph | 294 episodes, 1,241 entities, 1,610 relationship facts |

## Setup

```bash
git clone <repository-url>
cd agentic-rag-knowledge-graph

conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt

cp .env.example .env   # then fill in DATABASE_URL, NEO4J_*, LLM_*, EMBEDDING_* keys
```

Initialise the schema, then ingest:

```bash
# Schema (creates documents, chunks, page_embeddings, sessions, messages)
psql "$DATABASE_URL" -f sql/schema.sql

# Fast pipeline — embeddings only, 5-10 min
python scripts/ingest_fast.py --documents documents/

# Knowledge graph — restartable overnight run
python scripts/ingest_graph.py --limit 50

# Start the API
python -m agent.api          # http://localhost:8058
python cli.py                # interactive CLI (second terminal)
```

See `.env.example` for all configuration options. Sentry (`SENTRY_DSN`) and PageIndex (`PAGEINDEX_API_KEY`) are optional — the app runs without them.

## Security & Guardrails

- **Per-IP rate limiting** — 20 requests/hour/IP on public endpoints (429 + `Retry-After`); `/health` exempt
- **Input validation** — empty and >500-char queries rejected (400) before any LLM/DB work
- **Injection-resistant context** — retrieved document text is passed as tool-result data, never concatenated into the system prompt, so embedded instructions can't be elevated to the instruction plane

*Not implemented (stated honestly):* endpoint auth, output moderation/PII filtering, cross-instance shared rate limit. Add these before use beyond a demo.

## Observability

- **Metrics** — Prometheus at `/metrics`, including `agent_tool_invocations_total{tool_name=...}` (per-tool routing counter)
- **Error tracking** — Sentry, active only when `SENTRY_DSN` is set

## MCP Server

`mcp_server.py` exposes `vector_search`, `hybrid_search`, and `graph_search` over the Model Context Protocol via FastMCP — a thin wrapper reusing the same `agent/tools.py` implementations. See the file header for Claude Desktop config.

## Frontend

`frontend/` is a minimal Next.js + TypeScript UI: a single page with a query box that streams the agent's answer live from `/chat/stream` (Server-Sent Events) and displays colored pills for each tool the agent invoked (vector, hybrid, graph, PageIndex, etc.) — the retrieval routing is the point, so it's shown, not hidden.

**Status:** it runs locally against the deployed backend. It is **not** deployed to Vercel (or anywhere) yet — there is no live UI URL, only the backend's own `/docs`.

To run it:

```bash
cd frontend
npm install   # first time only
npm run dev   # http://localhost:3000, talks to the deployed backend
```

## API Reference

Interactive Swagger docs at `/docs` when the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Postgres + Neo4j connection status |
| `/chat`, `/chat/stream` | POST | Agent response (non-streaming / SSE) |
| `/search/{vector,graph,hybrid}` | POST | Direct retrieval (bypasses agent) |
| `/documents` | GET | List ingested documents |
| `/metrics` | GET | Prometheus metrics |

## Known Limitations

- **Graph coverage is partial** — populated but not every chunk is graph-enriched; graph tools may return sparse results for un-processed topics
- **No automated evaluation** — retrieval/answer quality tested manually; Ragas instrumentation is future work
- **Groq free-tier token limits** — heavy graph ingestion exhausts the daily allowance; use `--limit`

## Running Tests

```bash
pytest
pytest --cov=agent --cov=ingestion --cov-report=html
```

## License

MIT — see [LICENSE](LICENSE).