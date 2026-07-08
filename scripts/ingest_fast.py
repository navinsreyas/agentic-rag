#!/usr/bin/env python3
"""
ingest_fast.py — Vector ingestion only (PostgreSQL + page embeddings, NO knowledge graph).

Reads PDF and markdown files, chunks them, generates OpenAI embeddings, and
saves everything to the PostgreSQL ``documents``, ``chunks``, and
``page_embeddings`` tables.

NO Graphiti. NO Neo4j. NO LLM calls. Embedding API only.
Run ``ingest_graph.py`` separately when you have sufficient LLM credits.

Usage:
    python scripts/ingest_fast.py [--documents FOLDER] [--clean] [--dry-run] [--yes]
    python scripts/ingest_fast.py --force-pages          # backfill page_embeddings only

Flags:
    --dry-run      Scan files and print cost estimate, then exit (no DB writes)
    --yes / -y     Skip the confirmation prompt and start immediately
    --clean        Wipe existing PostgreSQL data before ingesting
    --force-pages  Re-ingest page embeddings for PDFs already in the DB,
                   without touching document/chunk records.  Use when
                   page_embeddings is empty but documents are already ingested.
    --verbose      Debug-level logging
"""

import os
import sys
import asyncio
import logging
import json
import glob
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import argparse

# On Windows the default console code page (cp1252) cannot encode the Unicode
# glyphs this script prints (→, ✓, ✗), which otherwise crashes with
# UnicodeEncodeError. Force UTF-8 output where supported.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Path setup — add project root to sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

from agent.db_utils import initialize_database, close_database, db_pool
from ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from ingestion.embedder import create_embedder, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants (OpenAI prices, per 1 M tokens, as of early 2025)
# ---------------------------------------------------------------------------
_PRICE_PER_MILLION: Dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
}
_CHARS_PER_TOKEN = 4


# ===========================================================================
# Pre-flight cost estimator  (no API calls, no DB connection required)
# ===========================================================================

def _count_pdf_pages(path: str) -> int:
    try:
        with open(path, "rb") as fh:
            return len(PyPDF2.PdfReader(fh).pages)
    except Exception:
        return 0


def _read_pdf_total_chars(path: str) -> Tuple[int, int]:
    try:
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            n_pages = len(reader.pages)
            chars = sum(
                len((p.extract_text() or "").strip())
                for p in reader.pages
            )
        return n_pages, chars
    except Exception:
        return 0, 0


def _read_text_chars(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return len(fh.read())
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as fh:
            return len(fh.read())
    except Exception:
        return 0


def _find_all_files(documents_folder: str) -> Tuple[List[str], List[str]]:
    """Return (md_files, pdf_files) lists."""
    if not os.path.exists(documents_folder):
        return [], []

    md_files: List[str] = []
    for pat in ("*.md", "*.markdown", "*.txt"):
        md_files.extend(
            glob.glob(os.path.join(documents_folder, "**", pat), recursive=True)
        )

    pdf_files: List[str] = []
    if _PYPDF2_AVAILABLE:
        pdf_files = glob.glob(
            os.path.join(documents_folder, "**", "*.pdf"), recursive=True
        )

    return sorted(md_files), sorted(pdf_files)


def estimate_cost(
    documents_folder: str,
    chunk_size: int = 1000,
    model: str = EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """Scan all files and estimate embedding API usage WITHOUT making any calls."""
    md_files, pdf_files = _find_all_files(documents_folder)
    all_files = md_files + pdf_files

    price_per_million = _PRICE_PER_MILLION.get(model, 0.020)

    file_rows: List[Dict[str, Any]] = []
    total_chunks  = 0
    total_pages   = 0
    total_chars   = 0

    for fp in all_files:
        is_pdf = fp.lower().endswith(".pdf")
        if is_pdf:
            n_pages, chars = _read_pdf_total_chars(fp)
        else:
            n_pages = 0
            chars   = _read_text_chars(fp)

        est_chunks = max(1, math.ceil(chars / chunk_size))
        chunk_tokens = est_chunks * (chunk_size // _CHARS_PER_TOKEN)
        page_chars   = chars if is_pdf else 0
        page_tokens  = math.ceil(page_chars / _CHARS_PER_TOKEN)
        total_est_tokens = chunk_tokens + page_tokens

        file_rows.append({
            "path":       fp,
            "name":       os.path.basename(fp),
            "is_pdf":     is_pdf,
            "pages":      n_pages,
            "chars":      chars,
            "est_chunks": est_chunks,
            "est_tokens": total_est_tokens,
        })

        total_chunks += est_chunks
        total_pages  += n_pages
        total_chars  += chars

    total_tokens = math.ceil(total_chars / _CHARS_PER_TOKEN) + (
        total_chunks * (chunk_size // _CHARS_PER_TOKEN)
    )
    estimated_cost = (total_tokens / 1_000_000) * price_per_million

    batch_size       = 100
    chunk_api_calls  = math.ceil(total_chunks / batch_size)
    page_api_calls   = len([r for r in file_rows if r["is_pdf"] and r["pages"] > 0])
    total_api_calls  = chunk_api_calls + page_api_calls

    return {
        "files":            file_rows,
        "total_files":      len(all_files),
        "total_chunks":     total_chunks,
        "total_pages":      total_pages,
        "total_tokens":     total_tokens,
        "total_api_calls":  total_api_calls,
        "chunk_api_calls":  chunk_api_calls,
        "page_api_calls":   page_api_calls,
        "estimated_cost":   estimated_cost,
        "model":            model,
        "price_per_million": price_per_million,
    }


def print_cost_estimate(est: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("EMBEDDING COST ESTIMATE  (no API calls made yet)")
    print("=" * 60)
    print(f"  Model            : {est['model']}")
    print(f"  Price            : ${est['price_per_million']:.3f} per 1M tokens")
    print()
    print(f"  {'File':<38} {'Chunks':>6}  {'Pages':>5}  {'~Tokens':>8}")
    print(f"  {'-'*38} {'-'*6}  {'-'*5}  {'-'*8}")
    for r in est["files"]:
        pages_str = str(r["pages"]) if r["is_pdf"] else "-"
        print(
            f"  {r['name']:<38} {r['est_chunks']:>6}  {pages_str:>5}  "
            f"{r['est_tokens']:>8,}"
        )
    print(f"  {'-'*38} {'-'*6}  {'-'*5}  {'-'*8}")
    print(
        f"  {'TOTAL':<38} {est['total_chunks']:>6}  "
        f"{est['total_pages']:>5}  {est['total_tokens']:>8,}"
    )
    print()
    print(f"  Embedding API calls  : {est['total_api_calls']}")
    print(f"    chunk batches      : {est['chunk_api_calls']}  (100 chunks per call)")
    print(f"    page batches       : {est['page_api_calls']}  (one per PDF)")
    print()
    print(f"  *** Estimated cost   : ${est['estimated_cost']:.4f} ***")
    print()
    print("  Note: already-ingested documents are skipped automatically.")
    print("  Note: token estimates use 4 chars/token -- actual may differ +/-20%.")
    print("=" * 60)


def estimate_pages_cost(
    documents_folder: str,
    model: str = EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """Estimate cost for --force-pages mode: page embeddings only."""
    _, pdf_files = _find_all_files(documents_folder)
    price_per_million = _PRICE_PER_MILLION.get(model, 0.020)

    file_rows: List[Dict[str, Any]] = []
    total_pages  = 0
    total_tokens = 0

    for fp in pdf_files:
        n_pages, chars = _read_pdf_total_chars(fp)
        tokens = math.ceil(chars / _CHARS_PER_TOKEN)
        file_rows.append({
            "name":   os.path.basename(fp),
            "pages":  n_pages,
            "tokens": tokens,
        })
        total_pages  += n_pages
        total_tokens += tokens

    total_api_calls = len(file_rows)
    estimated_cost  = (total_tokens / 1_000_000) * price_per_million

    return {
        "files":           file_rows,
        "total_pdfs":      len(pdf_files),
        "total_pages":     total_pages,
        "total_tokens":    total_tokens,
        "total_api_calls": total_api_calls,
        "estimated_cost":  estimated_cost,
        "model":           model,
        "price_per_million": price_per_million,
    }


def print_pages_cost_estimate(est: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("PAGE EMBEDDINGS COST ESTIMATE  (--force-pages)")
    print("=" * 60)
    print(f"  Model   : {est['model']}")
    print(f"  Price   : ${est['price_per_million']:.3f} per 1M tokens")
    print(f"  Mode    : page embeddings only -- no chunk re-embedding")
    print()
    print(f"  {'PDF file':<38} {'Pages':>5}  {'~Tokens':>8}")
    print(f"  {'-'*38} {'-'*5}  {'-'*8}")
    for r in est["files"]:
        print(f"  {r['name']:<38} {r['pages']:>5}  {r['tokens']:>8,}")
    print(f"  {'-'*38} {'-'*5}  {'-'*8}")
    print(f"  {'TOTAL':<38} {est['total_pages']:>5}  {est['total_tokens']:>8,}")
    print()
    print(f"  Embedding API calls : {est['total_api_calls']}  (one batch per PDF)")
    print()
    print(f"  *** Estimated cost  : ${est['estimated_cost']:.4f} ***")
    print()
    print("  Already-stored page rows are overwritten (ON CONFLICT DO UPDATE).")
    print("  PDFs not found in the documents table are skipped.")
    print("=" * 60)


# ===========================================================================
# Fast ingestion pipeline  (PostgreSQL + page_embeddings only)
# ===========================================================================

class FastIngestionPipeline:
    """
    Reads documents, chunks, embeds, and writes to PostgreSQL.
    Zero Graphiti / Neo4j / LLM calls — embedding API only.
    """

    def __init__(
        self,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_semantic_chunking: bool = True,
    ):
        self.documents_folder     = documents_folder
        self.clean_before_ingest  = clean_before_ingest

        chunker_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_splitting=use_semantic_chunking,
        )
        self.chunker  = create_chunker(chunker_config)
        self.embedder = create_embedder()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info("Connecting to PostgreSQL …")
        await initialize_database()
        self._initialized = True
        logger.info("Database ready")

    async def close(self) -> None:
        if self._initialized:
            await close_database()
            self._initialized = False

    async def run(self) -> List[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_postgres()

        md_files, pdf_files = _find_all_files(self.documents_folder)
        all_files = md_files + pdf_files

        if not all_files:
            logger.warning(f"No documents found in '{self.documents_folder}'")
            return []

        logger.info(
            f"Found {len(md_files)} markdown/text and {len(pdf_files)} PDF file(s)"
        )

        results: List[Dict[str, Any]] = []
        for i, fp in enumerate(all_files):
            label = f"[{i+1}/{len(all_files)}]"
            print(f"\n{label} {os.path.basename(fp)}")
            try:
                if fp.lower().endswith(".pdf"):
                    result = await self._ingest_pdf(fp)
                else:
                    result = await self._ingest_markdown(fp)
                results.append(result)
                if result.get("skipped"):
                    print(f"  → already in DB, skipped")
                elif result.get("error"):
                    print(f"  → ERROR: {result['error']}")
                else:
                    pages_str = (
                        f", {result['pages']} page embeddings"
                        if result.get("pages") else ""
                    )
                    print(f"  → {result['chunks']} chunks{pages_str}")
            except Exception as exc:
                logger.error(f"  Unhandled error: {exc}")
                results.append({
                    "title": os.path.basename(fp),
                    "chunks": 0, "pages": 0,
                    "skipped": False, "error": str(exc),
                })

        return results

    def _read_pdf_pages(self, file_path: str) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((idx + 1, text))
        return pages

    def _extract_pdf_title(self, file_path: str, first_page_text: str) -> str:
        try:
            with open(file_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                info = reader.metadata
                if info and getattr(info, "title", None) and info.title.strip():
                    return info.title.strip()
        except Exception:
            pass
        for line in first_page_text.split("\n")[:10]:
            line = line.strip()
            if line and len(line) < 120:
                return line
        return os.path.splitext(os.path.basename(file_path))[0]

    async def _ingest_pdf(self, file_path: str) -> Dict[str, Any]:
        start = datetime.now()

        try:
            pages = self._read_pdf_pages(file_path)
        except Exception as exc:
            return {
                "title": os.path.basename(file_path),
                "chunks": 0, "pages": 0, "skipped": False,
                "error": f"Could not read PDF: {exc}",
            }

        if not pages:
            return {
                "title": os.path.basename(file_path),
                "chunks": 0, "pages": 0, "skipped": False,
                "error": "No extractable text found in PDF",
            }

        title  = self._extract_pdf_title(file_path, pages[0][1])
        source = os.path.relpath(file_path, self.documents_folder)

        if await self._document_exists(title):
            return {"title": title, "chunks": 0, "pages": 0, "skipped": True, "error": None}

        metadata = {
            "file_path":      file_path,
            "file_size":      os.path.getsize(file_path),
            "page_count":     len(pages),
            "ingestion_date": datetime.now().isoformat(),
        }
        full_content = "\n\n".join(text for _, text in pages)

        all_chunks: List[DocumentChunk] = []
        for page_number, page_text in pages:
            page_chunks = await self.chunker.chunk_document(
                content=page_text,
                title=title,
                source=source,
                metadata={**metadata, "page_number": page_number},
            )
            all_chunks.extend(page_chunks)

        for idx, chunk in enumerate(all_chunks):
            chunk.index = idx

        if not all_chunks:
            return {
                "title": title, "chunks": 0, "pages": len(pages),
                "skipped": False, "error": "No chunks created",
            }

        print(f"  Embedding {len(all_chunks)} chunks …", end="", flush=True)
        embedded_chunks = await self.embedder.embed_chunks(all_chunks)
        print(" done")

        document_id = await self._save_to_postgres(
            title, source, full_content, embedded_chunks, metadata
        )

        pages_stored = 0
        try:
            from page_index_semantic import ingest_page_embeddings
            print(f"  Embedding {len(pages)} full pages …", end="", flush=True)
            pages_stored = await ingest_page_embeddings(
                document_id=document_id,
                pages=pages,
                doc_title=title,
                source=source,
                embedder=self.embedder,
            )
            print(" done")
        except Exception as exc:
            print(f" FAILED (non-fatal): {exc}")

        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"  PDF done in {elapsed:.1f}s")
        return {
            "title":   title,
            "chunks":  len(embedded_chunks),
            "pages":   pages_stored,
            "skipped": False,
            "error":   None,
        }

    def _read_text(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as fh:
                return fh.read()

    def _extract_md_title(self, content: str, file_path: str) -> str:
        for line in content.split("\n")[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return os.path.splitext(os.path.basename(file_path))[0]

    async def _ingest_markdown(self, file_path: str) -> Dict[str, Any]:
        start   = datetime.now()
        content = self._read_text(file_path)
        title   = self._extract_md_title(content, file_path)
        source  = os.path.relpath(file_path, self.documents_folder)
        metadata = {
            "file_path":      file_path,
            "file_size":      len(content),
            "ingestion_date": datetime.now().isoformat(),
            "word_count":     len(content.split()),
        }

        if await self._document_exists(title):
            return {"title": title, "chunks": 0, "pages": 0, "skipped": True, "error": None}

        chunks = await self.chunker.chunk_document(
            content=content, title=title, source=source, metadata=metadata,
        )
        if not chunks:
            return {
                "title": title, "chunks": 0, "pages": 0,
                "skipped": False, "error": "No chunks created",
            }

        print(f"  Embedding {len(chunks)} chunks …", end="", flush=True)
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        print(" done")

        await self._save_to_postgres(title, source, content, embedded_chunks, metadata)

        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"  Markdown done in {elapsed:.1f}s")
        return {
            "title": title, "chunks": len(embedded_chunks),
            "pages": 0, "skipped": False, "error": None,
        }

    async def _document_exists(self, title: str) -> bool:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM documents WHERE title = $1 LIMIT 1", title
            )
            return row is not None

    async def _get_document_id_by_title(self, title: str) -> Optional[str]:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id::text FROM documents WHERE title = $1 LIMIT 1", title
            )
            return row["id"] if row else None

    async def run_force_pages(self) -> List[Dict[str, Any]]:
        """Re-ingest page embeddings for every PDF without re-chunking documents."""
        if not self._initialized:
            await self.initialize()

        _, pdf_files = _find_all_files(self.documents_folder)
        if not pdf_files:
            print("No PDF files found in the documents folder.")
            return []

        from page_index_semantic import ingest_page_embeddings

        results: List[Dict[str, Any]] = []
        for i, fp in enumerate(pdf_files):
            label = f"[{i+1}/{len(pdf_files)}]"
            name  = os.path.basename(fp)
            print(f"\n{label} {name}")

            try:
                pages = self._read_pdf_pages(fp)
            except Exception as exc:
                print(f"  ERROR reading PDF: {exc}")
                results.append({"title": name, "pages": 0, "error": str(exc)})
                continue

            if not pages:
                print("  No extractable text — skipping")
                results.append({"title": name, "pages": 0, "error": "No extractable text"})
                continue

            title  = self._extract_pdf_title(fp, pages[0][1])
            source = os.path.relpath(fp, self.documents_folder)

            document_id = await self._get_document_id_by_title(title)
            if document_id is None:
                print(f"  '{title}' not found in documents table — skipping")
                print("  (Run without --force-pages to fully ingest this document first)")
                results.append({"title": title, "pages": 0, "error": "Not in documents table"})
                continue

            print(
                f"  doc_id {document_id[:8]}...  "
                f"Embedding {len(pages)} pages ...",
                end="", flush=True,
            )
            try:
                stored = await ingest_page_embeddings(
                    document_id=document_id,
                    pages=pages,
                    doc_title=title,
                    source=source,
                    embedder=self.embedder,
                )
                print(f" {stored} stored")
                results.append({"title": title, "pages": stored, "error": None})
            except Exception as exc:
                print(f" FAILED: {exc}")
                results.append({"title": title, "pages": 0, "error": str(exc)})

        return results

    async def _save_to_postgres(
        self,
        title:    str,
        source:   str,
        content:  str,
        chunks:   List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                doc_row = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title, source, content, json.dumps(metadata),
                )
                document_id = doc_row["id"]

                for chunk in chunks:
                    emb_str = None
                    if hasattr(chunk, "embedding") and chunk.embedding:
                        emb_str = "[" + ",".join(map(str, chunk.embedding)) + "]"
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id, chunk.content, emb_str,
                        chunk.index, json.dumps(chunk.metadata), chunk.token_count,
                    )

        return document_id

    async def _clean_postgres(self) -> None:
        logger.warning("Cleaning PostgreSQL tables …")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM page_embeddings")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        logger.info("PostgreSQL tables cleared")


# ===========================================================================
# Entry point
# ===========================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast vector ingestion — embeddings only, NO knowledge graph. "
            "Run scripts/ingest_graph.py separately to build the knowledge graph."
        )
    )
    parser.add_argument(
        "--documents", "-d", default="documents",
        help="Folder containing documents (default: documents)",
    )
    parser.add_argument(
        "--clean", "-c", action="store_true",
        help="Wipe all existing PostgreSQL data before ingesting",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate and exit without making any API calls or DB writes",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt and start immediately",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Overlap between consecutive chunks (default: 200)",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Use simple chunking instead of section-aware semantic chunking",
    )
    parser.add_argument(
        "--force-pages", action="store_true",
        help=(
            "Re-ingest page embeddings for PDFs already in the database, "
            "without re-chunking or re-embedding documents/chunks."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not _PYPDF2_AVAILABLE:
        print("WARNING: PyPDF2 not installed — PDF files will be skipped. (pip install PyPDF2)")

    # --force-pages branch
    if args.force_pages:
        print(f"\nScanning '{args.documents}' for PDFs (--force-pages mode) …")
        pages_est = estimate_pages_cost(documents_folder=args.documents, model=EMBEDDING_MODEL)

        if pages_est["total_pdfs"] == 0:
            print(f"No PDF files found in '{args.documents}'. Nothing to do.")
            return

        print_pages_cost_estimate(pages_est)

        if args.dry_run:
            print("--dry-run mode: exiting without making any changes.")
            return

        if not args.yes:
            try:
                answer = input("Proceed with page embedding ingestion? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return
            if answer not in ("y", "yes"):
                print("Aborted.")
                return

        pipeline = FastIngestionPipeline(documents_folder=args.documents)
        try:
            wall_start = datetime.now()
            results    = await pipeline.run_force_pages()
            elapsed    = (datetime.now() - wall_start).total_seconds()

            ok      = [r for r in results if not r.get("error")]
            errored = [r for r in results if r.get("error")]

            print("\n" + "=" * 60)
            print("FORCE-PAGES COMPLETE")
            print("=" * 60)
            print(f"  PDFs processed   : {len(ok)}")
            print(f"  Errors / skipped : {len(errored)}")
            print(f"  Page rows stored : {sum(r.get('pages', 0) for r in ok)}")
            print(f"  Wall time        : {elapsed:.1f}s")
            print()
            for r in results:
                if r.get("error"):
                    print(f"  ✗ {r['title']}  — {r['error']}")
                else:
                    print(f"  ✓ {r['title']}  — {r['pages']} page embeddings")
            print()
            print("PageIndex (semantic) is now ready.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            await pipeline.close()
        return

    # Normal ingestion branch
    print(f"\nScanning '{args.documents}' …")
    est = estimate_cost(
        documents_folder=args.documents,
        chunk_size=args.chunk_size,
        model=EMBEDDING_MODEL,
    )

    if est["total_files"] == 0:
        print(f"No documents found in '{args.documents}'. Nothing to do.")
        return

    print_cost_estimate(est)

    if args.dry_run:
        print("--dry-run mode: exiting without making any changes.")
        return

    if not args.yes:
        try:
            answer = input("Proceed with ingestion? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    pipeline = FastIngestionPipeline(
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
    )

    try:
        wall_start = datetime.now()
        results    = await pipeline.run()
        elapsed    = (datetime.now() - wall_start).total_seconds()

        ingested = [r for r in results if not r.get("skipped") and not r.get("error")]
        skipped  = [r for r in results if r.get("skipped")]
        errored  = [r for r in results if r.get("error")]

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"  Ingested         : {len(ingested)} document(s)")
        print(f"  Skipped (dup)    : {len(skipped)}")
        print(f"  Errors           : {len(errored)}")
        print(f"  Total chunks     : {sum(r.get('chunks', 0) for r in ingested)}")
        print(f"  Page embeddings  : {sum(r.get('pages', 0) for r in ingested)}")
        print(f"  Wall time        : {elapsed:.1f}s")
        print()

        for r in results:
            if r.get("error"):
                print(f"  ✗ {r['title']}  — {r['error']}")
            elif r.get("skipped"):
                print(f"  ~ {r['title']}  [already in DB]")
            else:
                pages_str = f", {r['pages']} page embeddings" if r.get("pages") else ""
                print(f"  ✓ {r['title']}  — {r['chunks']} chunks{pages_str}")

        print()
        print("Vector search and PageIndex are ready.")
        if errored:
            print(f"WARNING: {len(errored)} document(s) failed — check logs above.")
        print("Run  python scripts/ingest_graph.py --limit 50  when you have LLM credits.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
