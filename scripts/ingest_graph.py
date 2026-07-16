#!/usr/bin/env python3
"""
ingest_graph.py — Knowledge graph ingestion from existing PostgreSQL chunks.

Reads chunks already stored in the ``chunks`` table, adds them to the
Graphiti / Neo4j knowledge graph, and tracks which chunks have been processed
in ``graph_progress.json`` so the script can be stopped and restarted safely
without duplicating work.

Usage:
    python scripts/ingest_graph.py [--limit N] [--verbose]

Examples:
    python scripts/ingest_graph.py --limit 50    # process the next 50 unprocessed chunks
    python scripts/ingest_graph.py               # process ALL remaining unprocessed chunks

Run ``scripts/ingest_fast.py`` first to populate the PostgreSQL tables.
"""

import os
import sys
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import argparse

# On Windows the default console code page (cp1252) cannot encode the Unicode
# glyphs this script prints (→, ✓, ✗), which otherwise crashes with
# UnicodeEncodeError. Force UTF-8 output where supported.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Path setup — add project root to sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from agent.db_utils import initialize_database, close_database, db_pool
from agent.graph_utils import GraphitiClient

logger = logging.getLogger(__name__)

PROGRESS_FILE = os.path.join(_ROOT, "graph_progress.json")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> set:
    """Return the set of chunk IDs already added to the knowledge graph."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(data.get("processed_chunk_ids", []))
    except Exception as exc:
        logger.warning(f"Could not read progress file: {exc} — starting fresh")
        return set()


def save_progress(processed_ids: set) -> None:
    """Persist the set of processed chunk IDs to graph_progress.json."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "processed_chunk_ids": list(processed_ids),
                "last_updated": datetime.now().isoformat(),
                "count": len(processed_ids),
            },
            fh,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

_MAX_EPISODE_CHARS = 6000


def _prepare_episode_content(content: str, title: str) -> str:
    """Truncate content to fit Graphiti token limits and prepend the doc title."""
    if len(content) > _MAX_EPISODE_CHARS:
        truncated = content[:_MAX_EPISODE_CHARS]
        cut = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
        )
        if cut > _MAX_EPISODE_CHARS * 0.7:
            content = truncated[: cut + 1] + " [TRUNCATED]"
        else:
            content = truncated + "... [TRUNCATED]"

    if title and len(content) < _MAX_EPISODE_CHARS - 100:
        return f"[Doc: {title[:50]}]\n\n{content}"
    return content


def _build_episode_id(title: str, page_number: Any, chunk_index: int) -> str:
    """Build episode ID: {safe_title}_p{page_number}_chunk{index}"""
    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title[:40])
    page_suffix = f"_p{page_number}" if page_number else ""
    return f"{safe_title}{page_suffix}_chunk{chunk_index}"


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------

async def fetch_unprocessed_chunks(
    processed_ids: set,
    limit: Optional[int],
) -> List[Any]:
    """Fetch chunks from PostgreSQL that have not yet been added to Graphiti."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id::text      AS chunk_id,
                c.content,
                c.chunk_index,
                c.metadata,
                d.title         AS doc_title,
                d.source        AS doc_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY d.title, c.chunk_index
            """
        )

    unprocessed = [r for r in rows if r["chunk_id"] not in processed_ids]

    if limit is not None:
        unprocessed = unprocessed[:limit]

    return unprocessed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Graph ingestion: PostgreSQL chunks → Graphiti / Neo4j. "
            "Tracks progress in graph_progress.json so you can stop and restart safely."
        )
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Maximum chunks to process in this run (default: all remaining)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    processed_ids = load_progress()
    logger.info(f"Already in graph: {len(processed_ids)} chunk(s)")

    logger.info("Connecting to PostgreSQL …")
    await initialize_database()

    chunks = await fetch_unprocessed_chunks(processed_ids, args.limit)
    total = len(chunks)

    if total == 0:
        remaining_msg = (
            f"All {len(processed_ids)} ingested chunk(s) are already in the graph."
            if processed_ids
            else "No chunks found in PostgreSQL. Run scripts/ingest_fast.py first."
        )
        print(remaining_msg)
        await close_database()
        return

    limit_msg = f" (limit: {args.limit})" if args.limit else ""
    logger.info(f"Chunks to process this run: {total}{limit_msg}")

    logger.info("Connecting to Graphiti / Neo4j …")
    graph = GraphitiClient()
    await graph.initialize()

    episodes_added = 0
    errors = 0
    wall_start = datetime.now()

    try:
        for i, row in enumerate(chunks):
            chunk_id    = row["chunk_id"]
            content     = row["content"]
            chunk_index = row["chunk_index"]
            doc_title   = row["doc_title"]
            doc_source  = row["doc_source"]

            raw_meta = row["metadata"]
            metadata: Dict[str, Any] = (
                json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            )
            page_number = metadata.get("page_number", "")

            episode_id      = _build_episode_id(doc_title, page_number, chunk_index)
            episode_content = _prepare_episode_content(content, doc_title)

            if page_number:
                source_desc = (
                    f"Document: {doc_title} | Page: {page_number} | Chunk: {chunk_index}"
                )
            else:
                source_desc = f"Document: {doc_title} (Chunk: {chunk_index})"

            try:
                await graph.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_desc,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": doc_title,
                        "document_source": doc_source,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                )
                episodes_added += 1
                processed_ids.add(chunk_id)
                save_progress(processed_ids)
                logger.info(f"[{i+1}/{total}] ✓ {episode_id}")

            except Exception as exc:
                logger.error(f"[{i+1}/{total}] ✗ chunk {chunk_id}: {exc}")
                errors += 1
                processed_ids.add(chunk_id)
                save_progress(processed_ids)

            if i < total - 1:
                await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Re-run to continue where you left off.")

    finally:
        await graph.close()
        await close_database()

    elapsed = (datetime.now() - wall_start).total_seconds()

    print("\n" + "=" * 55)
    print("GRAPH INGESTION SUMMARY")
    print("=" * 55)
    print(f"  Episodes added  : {episodes_added}")
    print(f"  Errors          : {errors}")
    print(f"  Total in graph  : {len(processed_ids)}")
    print(f"  Wall time       : {elapsed:.1f}s")

    try:
        await initialize_database()
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS n FROM chunks")
            grand_total = row["n"]
        remaining = grand_total - len(processed_ids)
        if remaining > 0:
            print(f"  Still remaining : {remaining} chunk(s)")
            print(f"  Next run        : python scripts/ingest_graph.py --limit {min(remaining, 50)}")
        else:
            print("  Knowledge graph is fully up to date.")
        await close_database()
    except Exception:
        pass

    print()


if __name__ == "__main__":
    asyncio.run(main())
