"""
FastMCP server exposing this project's retrieval tools over the Model Context
Protocol (MCP), so MCP clients (e.g. Claude Desktop) can call them directly.

Design: this is a THIN WRAPPER. It imports and calls the exact same tool
implementations the FastAPI app uses (`agent.tools.*`) against the same
databases (Neon Postgres + Neo4j Aura via `.env`). There is one source of truth
for retrieval logic — this file adds no retrieval logic of its own, only MCP
tool registration and result serialization.

Run standalone (stdio transport, what Claude Desktop uses):
    python mcp_server.py
"""

import os
import sys
import asyncio
from typing import Any, Dict, List

# --- Make this runnable regardless of the client's working directory ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Load .env from THIS repo (explicit path) BEFORE importing agent.* — agent.graph_utils
# maps NEO4J_DATABASE -> DEFAULT_DATABASE at import time, so the env must be populated
# first or graph queries would target the wrong database.
from dotenv import load_dotenv
load_dotenv(os.path.join(_HERE, ".env"))

from mcp.server.fastmcp import FastMCP

# The SAME implementations the FastAPI app uses — no reimplementation.
from agent.db_utils import initialize_database
from agent.graph_utils import initialize_graph
from agent.tools import (
    vector_search_tool,
    hybrid_search_tool,
    graph_search_tool,
    VectorSearchInput,
    HybridSearchInput,
    GraphSearchInput,
)

mcp = FastMCP("agentic-rag")

# ---------------------------------------------------------------------------
# One-time, idempotent init of the shared DB connections (Neon + Aura).
# Reuses the app's connection setup so the MCP layer and the API share config.
# ---------------------------------------------------------------------------
_init_lock = asyncio.Lock()
_initialized = False


async def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        await initialize_database()   # Neon pool + match_pages() function
        await initialize_graph()      # Neo4j Aura / Graphiti client
        _initialized = True


@mcp.tool()
async def vector_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Semantic vector search over document chunks (pgvector cosine similarity).

    Use for broad or conceptual questions where meaning matters more than exact
    wording (e.g. "What is Constitutional AI?"). Returns the chunks whose
    embeddings are closest to the query.

    Args:
        query: Natural-language search query.
        limit: Maximum number of chunks to return (1-50; default 10).

    Returns:
        A list of matching chunks, each with content, similarity score, and the
        source document's title and path.
    """
    await _ensure_initialized()
    results = await vector_search_tool(VectorSearchInput(query=query, limit=limit))
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]


@mcp.tool()
async def hybrid_search(
    query: str, limit: int = 10, text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """Hybrid search: combines vector similarity with keyword (full-text) ranking.

    Use when both exact terms AND semantic meaning matter (specific names,
    acronyms, statistics). `text_weight` blends the two signals: 0.0 = pure
    vector, 1.0 = pure keyword.

    Args:
        query: Natural-language search query.
        limit: Maximum number of chunks to return (1-50; default 10).
        text_weight: Weight of the keyword signal vs. vector (0.0-1.0; default 0.3).

    Returns:
        A list of chunks ranked by the combined score, each with content, score,
        and the source document's title and path.
    """
    await _ensure_initialized()
    results = await hybrid_search_tool(
        HybridSearchInput(query=query, limit=limit, text_weight=text_weight)
    )
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]


@mcp.tool()
async def graph_search(query: str) -> List[Dict[str, Any]]:
    """Search the temporal knowledge graph (Neo4j/Graphiti) for facts & relationships.

    Use for questions about how named entities relate, or how facts changed over
    time (e.g. "How does OpenAI relate to Microsoft?"). Returns fact statements
    with their validity timestamps, not document chunks.

    Args:
        query: Natural-language query about entities/relationships.

    Returns:
        A list of facts, each with the fact text, its uuid, and valid_at /
        invalid_at timestamps when known.
    """
    await _ensure_initialized()
    results = await graph_search_tool(GraphSearchInput(query=query))
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "invalid_at": r.invalid_at,
            "source_node_uuid": r.source_node_uuid,
        }
        for r in results
    ]


if __name__ == "__main__":
    # stdio transport — this is what Claude Desktop launches and speaks.
    mcp.run()
