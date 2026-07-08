#!/usr/bin/env python3
"""
ingest_pageindex.py — Upload PDF documents to the PageIndex cloud API.

IMPORTANT — API REALITY NOTE:
The PageIndex library (v0.2.8) is a REST client for api.pageindex.ai, a cloud
service.  There are no local tree files.  This script uploads each PDF to the
PageIndex cloud, waits until processing is complete, then saves the returned
doc_id mappings in pageindex_trees/index.json so that the retrieval tool can
find them at query time.

Why doc_ids instead of tree JSON files?
  The tree is built on VectifyAI's servers.  We only hold a pointer (doc_id)
  to it.  Calling pageindex_search at query time sends the question to the
  cloud and gets back the relevant section text.

ADAPTATION FROM ORIGINAL SPEC:
  - "Builds a PageIndex tree for each document" → uploads PDF to cloud API
  - "Saves each tree as a JSON file" → saves doc_id in pageindex_trees/index.json
  - PAGEINDEX_MODEL env var does NOT apply — model selection is server-side.
    This script uses PAGEINDEX_API_KEY instead.
  - Markdown files are skipped — the PageIndex upload API only accepts PDFs.

Usage:
    python ingestion/ingest_pageindex.py [--documents FOLDER] [--dry-run] [--yes]

Flags:
    --dry-run    List documents that would be uploaded, then exit
    --yes / -y   Skip the confirmation prompt
    --documents  Folder to scan (default: documents)
    --verbose    Debug-level logging
"""

import os
import sys
import json
import glob
import asyncio
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Path setup — allow running directly from project root or ingestion/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TREES_DIR     = os.path.join(_ROOT, "pageindex_trees")
INDEX_FILE    = os.path.join(TREES_DIR, "index.json")
POLL_INTERVAL = 10   # seconds between readiness checks
POLL_TIMEOUT  = 300  # maximum seconds to wait per document


# ---------------------------------------------------------------------------
# Index helpers  (the index maps relative-path → {doc_id, name, uploaded_at})
# ---------------------------------------------------------------------------

def _load_index() -> Dict[str, Any]:
    """Load the existing doc_id index, or return an empty dict."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: Dict[str, Any]) -> None:
    """Persist the doc_id index to disk, creating the directory if needed."""
    os.makedirs(TREES_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def _find_pdfs(documents_folder: str) -> List[str]:
    return sorted(
        glob.glob(os.path.join(documents_folder, "**", "*.pdf"), recursive=True)
    )


def _find_markdowns(documents_folder: str) -> List[str]:
    md_files: List[str] = []
    for pat in ("*.md", "*.markdown", "*.txt"):
        md_files.extend(
            glob.glob(os.path.join(documents_folder, "**", pat), recursive=True)
        )
    return sorted(md_files)


# ---------------------------------------------------------------------------
# Upload + polling logic
# ---------------------------------------------------------------------------

async def _upload_and_wait(
    client,
    file_path: str,
    poll_interval: int = POLL_INTERVAL,
    timeout: int = POLL_TIMEOUT,
) -> Optional[str]:
    """
    Upload one PDF to the PageIndex API and poll until it is retrieval-ready.

    Returns the doc_id string on success, or None on failure / timeout.

    Why asyncio.to_thread?  The PageIndex client uses the 'requests' library
    (synchronous HTTP).  Wrapping each call in asyncio.to_thread lets us run
    the blocking I/O on a thread-pool without blocking the event loop.
    """
    logger.debug(f"Uploading {file_path} ...")
    try:
        result = await asyncio.to_thread(client.submit_document, file_path)
        doc_id = result.get("doc_id")
        if not doc_id:
            logger.error(f"No doc_id in API response: {result}")
            return None
    except Exception as exc:
        logger.error(f"Upload failed for {file_path}: {exc}")
        return None

    # Poll until the document tree is built on the server
    elapsed = 0
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        try:
            ready = await asyncio.to_thread(client.is_retrieval_ready, doc_id)
            if ready:
                return doc_id
            logger.debug(
                f"  {os.path.basename(file_path)}: still processing "
                f"({elapsed}s elapsed) ..."
            )
        except Exception as exc:
            logger.warning(f"  Readiness check failed: {exc}")

    logger.error(
        f"Timed out waiting for {os.path.basename(file_path)} "
        f"after {timeout}s.  Check your PageIndex dashboard."
    )
    return None


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

async def run_ingestion(
    documents_folder: str,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    # --- API key check ---------------------------------------------------
    api_key = os.environ.get("PAGEINDEX_API_KEY", "")
    if not api_key:
        print(
            "\nERROR: PAGEINDEX_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "  PAGEINDEX_API_KEY=pi_your_key_here\n"
            "Get an API key at https://pageindex.ai\n"
        )
        sys.exit(1)

    from pageindex import PageIndexClient
    client = PageIndexClient(api_key=api_key)

    # --- Scan files -------------------------------------------------------
    pdf_files = _find_pdfs(documents_folder)
    md_files  = _find_markdowns(documents_folder)

    if not pdf_files and not md_files:
        print(f"\nNo documents found in '{documents_folder}'.")
        return

    # --- Determine what to upload vs. skip --------------------------------
    index    = _load_index()
    to_upload: List[str] = []
    to_skip:   List[str] = []

    for fp in pdf_files:
        key = os.path.relpath(fp, _ROOT)
        if key in index:
            to_skip.append(fp)
        else:
            to_upload.append(fp)

    # --- Print plan -------------------------------------------------------
    print()
    print("=" * 64)
    print("PAGEINDEX UPLOAD PLAN")
    print("=" * 64)
    print(f"  Documents folder  : {documents_folder}")
    print(f"  PDF files found   : {len(pdf_files)}")

    if md_files:
        print(
            f"  Markdown files    : {len(md_files)}"
            f"  (skipped — PageIndex cloud API accepts PDFs only)"
        )

    print()
    print("  NOTE: PageIndex is a cloud API service (api.pageindex.ai).")
    print("  Tree building happens on VectifyAI's servers, not locally.")
    print("  Billing is managed via your PageIndex dashboard.")
    print("  PAGEINDEX_MODEL is not applicable — model is chosen server-side.")
    print()

    if to_skip:
        print(f"  Already uploaded ({len(to_skip)} — will skip):")
        for fp in to_skip:
            key    = os.path.relpath(fp, _ROOT)
            doc_id = index[key].get("doc_id", "?")
            print(
                f"    ~ {os.path.basename(fp):<50}"
                f"  doc_id: {doc_id[:16]}..."
            )
        print()

    if not to_upload:
        print("  Nothing new to upload. All PDFs are already indexed.")
        print("=" * 64)
        return

    print(f"  To upload ({len(to_upload)}):")
    total_bytes = 0
    for fp in to_upload:
        size_kb = os.path.getsize(fp) / 1024
        total_bytes += os.path.getsize(fp)
        print(f"    + {os.path.basename(fp):<50}  {size_kb:>6.0f} KB")

    print()
    print(
        f"  Total upload size  : {total_bytes / 1024:.0f} KB"
        f"  ({total_bytes / 1_048_576:.2f} MB)"
    )
    print(
        f"  Estimated wait     : ~{len(to_upload) * 30}–{len(to_upload) * 60}s "
        f"(processing time varies by document size)"
    )
    print("=" * 64)

    if dry_run:
        print("\n--dry-run mode: exiting without uploading anything.")
        return

    if not yes:
        try:
            answer = input("\nProceed with upload to PageIndex cloud? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # --- Upload -----------------------------------------------------------
    print()
    successes = []
    failures  = []

    for i, fp in enumerate(to_upload):
        label = f"[{i+1}/{len(to_upload)}]"
        name  = os.path.basename(fp)
        print(f"{label} {name}")
        print(f"  Uploading ...", end="", flush=True)

        doc_id = await _upload_and_wait(client, fp)

        if doc_id:
            key = os.path.relpath(fp, _ROOT)
            index[key] = {
                "doc_id":      doc_id,
                "name":        name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_index(index)
            print(f" done  → doc_id: {doc_id}")
            successes.append(name)
        else:
            print(f" FAILED  (see logs above for details)")
            failures.append(name)

    # --- Summary ----------------------------------------------------------
    print()
    print("=" * 64)
    print("UPLOAD COMPLETE")
    print("=" * 64)
    print(f"  Uploaded successfully : {len(successes)}")
    print(f"  Failed / timed out   : {len(failures)}")
    print(f"  Already indexed      : {len(to_skip)}")
    print(f"  Index file           : {INDEX_FILE}")
    print()

    if successes:
        for name in successes:
            print(f"  ✓ {name}")
    if failures:
        for name in failures:
            print(f"  ✗ {name}  (retry or check PageIndex dashboard)")

    print()
    print("PageIndex retrieval is now ready for uploaded documents.")
    print("Run the agent and use pageindex_search to query them.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload PDF documents to the PageIndex cloud API and save doc_id "
            "mappings to pageindex_trees/index.json for use by the agent."
        )
    )
    parser.add_argument(
        "--documents", "-d", default="documents",
        help="Folder containing documents (default: documents)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List documents that would be uploaded, then exit without uploading",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt and start immediately",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    asyncio.run(
        run_ingestion(
            documents_folder=args.documents,
            dry_run=args.dry_run,
            yes=args.yes,
        )
    )


if __name__ == "__main__":
    main()
