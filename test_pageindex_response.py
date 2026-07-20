#!/usr/bin/env python3
"""
test_pageindex_response.py

Step 1: Send a query to the Risk Report doc_id and print the FULL raw API
        response so we can see exactly what structure PageIndex returns.

Step 2: Print the extracted answer using the parser in agent/tools.py.

Run from the project root:
    python test_pageindex_response.py
"""

import json
import time
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

import os
from pageindex import PageIndexClient

API_KEY  = os.environ["PAGEINDEX_API_KEY"]

# Documents indexed in pageindex_trees/index.json
DOCS = {
    "risk_report":    ("pi-cmnhdbvtb001o01r016wy7v0t",  "Redacted Risk Report Feb 2026.pdf"),
    "anthropic_info": ("pi-cmnhcekgr0i0v01r4hbm0tn62",  "Anthropic-Research-info-sheet.pdf"),
    "openai_2024":    ("pi-cmnhd4ea1000n01r0ubziki9x",   "OpenAI-2024.pdf"),
}

# Query the Anthropic info sheet with content it actually contains.
# "Constitutional AI" is NOT in this document — use content that IS.
QUERIES = [
    ("anthropic_info", "What does the Interpretability team work on?"),
]

MAX_WAIT = 300  # seconds total before giving up (large PDFs can take 3-5 min)


def poll_until_done(client: PageIndexClient, retrieval_id: str) -> dict:
    """Block until the retrieval is finished and return the raw result dict.

    Polls immediately first (no initial sleep) in case the API responds very fast.
    Falls back to exponential-ish back-off up to MAX_WAIT seconds total.
    """
    elapsed = 0
    intervals = [0, 2, 3, 5, 5, 10, 10, 15, 15, 15, 20, 20, 20, 20, 20, 20, 30, 30, 30]
    for gap in intervals:
        time.sleep(gap)
        elapsed += gap
        try:
            result = client.get_retrieval(retrieval_id)
        except Exception as exc:
            print(f"  [{elapsed:>3}s] get_retrieval ERROR: {exc}")
            continue

        status = result.get("status", "")
        print(f"  [{elapsed:>3}s] status={status!r}  keys={list(result.keys())}")

        is_done = status in ("completed", "done", "finished") or (
            not status
            and any(k in result for k in ("answer", "text", "content", "result", "nodes"))
        )
        if is_done:
            return result
        if status == "failed":
            raise RuntimeError(f"Retrieval failed on server: {result}")

        if elapsed >= MAX_WAIT:
            break

    raise TimeoutError(f"Retrieval did not finish within {MAX_WAIT}s")


def main():
    client = PageIndexClient(api_key=API_KEY)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent.tools import _extract_retrieval_content

    # ------------------------------------------------------------------ #
    #  PARSER UNIT TEST using the confirmed live API schema               #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("PARSER UNIT TEST (confirmed schema from live API)")
    print("=" * 70)

    # Simulate a response with populated retrieved_nodes
    mock_with_nodes = {
        "retrieval_id": "sr-test",
        "doc_id": "pi-test",
        "status": "completed",
        "query": "What is Interpretability?",
        "retrieved_nodes": [
            {
                "title": "Interpretability",
                "node_id": "0003",
                "page_index": 1,
                "text": "### Interpretability\n\nAI Interpretability is about understanding how models actually work.",
            }
        ],
    }
    mock_empty_nodes = {
        "retrieval_id": "sr-empty",
        "doc_id": None,
        "status": "completed",
        "query": "What is Constitutional AI?",
        "retrieved_nodes": [],
    }

    result1 = _extract_retrieval_content(mock_with_nodes)
    result2 = _extract_retrieval_content(mock_empty_nodes)

    print(f"\n  populated nodes -> parser returned {len(result1)} chars")
    print(f"  preview: {result1[:120]!r}")
    assert "Interpretability" in result1, "FAIL: parser missed retrieved_nodes content"
    print("  PASS: correctly extracted text from retrieved_nodes")

    print(f"\n  empty nodes     -> parser returned {len(result2)} chars")
    assert result2 == "", f"FAIL: expected empty string, got: {result2!r}"
    print("  PASS: correctly returned empty string for empty retrieved_nodes")
    print()

    for doc_key, query in QUERIES:
        doc_id, doc_name = DOCS[doc_key]

        # ------------------------------------------------------------------ #
        #  PHASE 1 — submit the query and poll for the result                 #
        # ------------------------------------------------------------------ #
        print()
        print("=" * 70)
        print(f"QUERY: {query!r}")
        print(f"  doc  : {doc_name}")
        print(f"  id   : {doc_id}")
        print("=" * 70)

        submit_result = client.submit_query(doc_id, query)
        print(f"\nsubmit_query() raw response: {submit_result}\n")

        retrieval_id = submit_result.get("retrieval_id")
        if not retrieval_id:
            print("ERROR: no retrieval_id in submit response")
            continue

        print(f"Polling get_retrieval({retrieval_id!r}) ...")
        try:
            raw = poll_until_done(client, retrieval_id)
        except (TimeoutError, RuntimeError) as exc:
            print(f"FAILED: {exc}")
            continue

        # ------------------------------------------------------------------ #
        #  PHASE 2 — print the complete raw response, nothing hidden          #
        # ------------------------------------------------------------------ #
        print()
        print("--- pprint (structured) ---")
        pprint(raw, sort_dicts=False, width=100)
        print()
        print("--- json.dumps (exact serialisation) ---")
        print(json.dumps(raw, indent=2, default=str))

        # ------------------------------------------------------------------ #
        #  PHASE 3 — run the current parser and show what it extracts         #
        # ------------------------------------------------------------------ #
        print()
        print("--- _extract_retrieval_content() output ---")
        extracted = _extract_retrieval_content(raw)
        if extracted:
            print(extracted[:1500])
        else:
            print("(parser returned empty string — needs fixing)")
        print()


if __name__ == "__main__":
    main()
