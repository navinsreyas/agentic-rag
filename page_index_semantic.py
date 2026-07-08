"""
Semantic PageIndex — page-level retrieval using full-page vector embeddings.

Each PDF page is embedded as a complete unit and stored in the
``page_embeddings`` PostgreSQL table. At query time the query is embedded
and compared against page-level vectors using cosine similarity, returning
whole pages rather than sub-page chunks.

Unlike keyword/word-overlap page matching, this uses dense vector similarity:
it returns pages whose *meaning* is closest to the query. It backs the agent's
``page_vector_search`` tool.

The three public functions follow the interface described in the project
design doc:

    ingest_page_embeddings(document_id, pages, doc_title, source, embedder)
    semantic_page_search(query, embedder, top_n=3)
    build_page_search_function(db_pool)
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ingest — write full-page embeddings to the database
# ---------------------------------------------------------------------------


async def ingest_page_embeddings(
    document_id: str,
    pages: List[Tuple[int, str]],
    doc_title: str,
    source: str,
    embedder,
) -> int:
    """
    Embed each page's full text and store one row per page in ``page_embeddings``.

    Uses ``INSERT … ON CONFLICT DO UPDATE`` so re-ingesting the same document
    safely refreshes the stored text and vectors without creating duplicates.
    The unique constraint is on ``(document_id, page_number)``.

    Args:
        document_id: UUID string of the parent ``documents`` record.
        pages: List of ``(page_number, page_text)`` tuples.  Page numbers are
               1-based to match human-readable PDF references.
        doc_title: Human-readable document title stored alongside each row for
                   fast display without joining to ``documents``.
        source: Relative file path stored for traceability.
        embedder: An ``EmbeddingGenerator`` instance (from
                  ``ingestion.embedder``) with a
                  ``generate_embeddings_batch(texts)`` method.

    Returns:
        Number of pages successfully stored.
    """
    try:
        from agent.db_utils import db_pool
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent.db_utils import db_pool

    if not pages:
        return 0

    page_texts = [text for _, text in pages]

    try:
        embeddings = await embedder.generate_embeddings_batch(page_texts)
    except Exception as exc:
        logger.error(f"ingest_page_embeddings: embedding batch failed — {exc}")
        return 0

    stored = 0
    # Each page is written in its own statement (no wrapping transaction).
    # This means a failure on one page does not abort the inserts for the
    # remaining pages — each INSERT auto-commits independently.
    async with db_pool.acquire() as conn:
        for (page_number, page_text), embedding in zip(pages, embeddings):
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            try:
                await conn.execute(
                    """
                    INSERT INTO page_embeddings
                        (document_id, page_number, content, embedding, doc_title, source)
                    VALUES ($1::uuid, $2, $3, $4::vector, $5, $6)
                    ON CONFLICT (document_id, page_number)
                    DO UPDATE SET
                        content   = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        doc_title = EXCLUDED.doc_title,
                        source    = EXCLUDED.source
                    """,
                    document_id,
                    page_number,
                    page_text,
                    embedding_str,
                    doc_title,
                    source,
                )
                stored += 1
            except Exception as exc:
                # Print directly so the error is always visible regardless of
                # whether a log handler has been configured.
                print(
                    f"  [page_embeddings] ERROR page {page_number} "
                    f"of '{doc_title}': {exc}"
                )
                logger.error(
                    f"ingest_page_embeddings: failed to store page {page_number} "
                    f"of '{doc_title}' — {exc}"
                )

    return stored


# ---------------------------------------------------------------------------
# Search — embed query, retrieve top pages by cosine similarity
# ---------------------------------------------------------------------------


async def semantic_page_search(
    query: str,
    embedder,
    top_n: int = 3,
) -> str:
    """
    Find the most semantically similar pages to a query.

    Embeds the query using the provided embedder, then runs a vector
    similarity search against the ``page_embeddings`` table, returning the
    full text of the top-N pages.

    Args:
        query: The natural-language query to search for.
        embedder: Any object with an async ``generate_embedding(text)`` method,
                  OR an async callable ``(text) -> List[float]`` directly.
        top_n: Number of pages to return (default 3).

    Returns:
        A formatted string with a header for each page showing document name,
        page number, and similarity score, followed by the complete page text.
        Returns ``"No relevant pages found."`` when the table is empty or no
        pages pass the similarity threshold.
    """
    try:
        from agent.db_utils import page_vector_search
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent.db_utils import page_vector_search

    # Support both an embedder object (.generate_embedding) and a plain callable
    try:
        if callable(embedder) and not hasattr(embedder, "generate_embedding"):
            embedding = await embedder(query)
        else:
            embedding = await embedder.generate_embedding(query)
    except Exception as exc:
        logger.error(f"semantic_page_search: embedding failed — {exc}")
        return "Page index search failed (embedding error)."

    try:
        results = await page_vector_search(embedding, limit=top_n)
    except Exception as exc:
        logger.error(f"semantic_page_search: DB query failed — {exc}")
        return "Page index search failed (database error)."

    if not results:
        return "No relevant pages found."

    parts = []
    for r in results:
        header = (
            f"=== [{r['doc_title']}] Page {r['page_number']} "
            f"(similarity: {r['similarity']:.3f}) ==="
        )
        parts.append(f"{header}\n\n{r['content']}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Setup — ensure the match_pages SQL function exists
# ---------------------------------------------------------------------------


async def build_page_search_function(db_pool) -> None:
    """
    Create or replace the ``match_pages`` PostgreSQL function.

    Safe to call on every startup — uses ``CREATE OR REPLACE``.  Mirrors the
    function already in ``sql/schema.sql`` but can be applied without re-running
    the full schema (which drops all tables).

    Args:
        db_pool: An ``asyncpg`` connection pool (the global ``db_pool`` from
                 ``agent.db_utils`` works directly).
    """
    sql = """
    CREATE OR REPLACE FUNCTION match_pages(
        query_embedding vector(1536),
        match_count INT DEFAULT 3
    )
    RETURNS TABLE (
        page_id UUID,
        document_id UUID,
        page_number INTEGER,
        content TEXT,
        similarity FLOAT,
        doc_title TEXT,
        source TEXT
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            p.id          AS page_id,
            p.document_id,
            p.page_number,
            p.content,
            1 - (p.embedding <=> query_embedding) AS similarity,
            p.doc_title,
            p.source
        FROM page_embeddings p
        WHERE p.embedding IS NOT NULL
        ORDER BY p.embedding <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    """
    async with db_pool.acquire() as conn:
        await conn.execute(sql)
    logger.info("match_pages SQL function created/updated in database")
