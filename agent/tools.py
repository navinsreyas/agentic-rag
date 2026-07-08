"""
Tools for the Pydantic AI agent.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks
)
from .graph_utils import (
    search_knowledge_graph,
    get_entity_relationships,
    graph_client
)
from .models import ChunkResult, GraphSearchResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize embedding client with flexible provider
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector
    """
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class GraphSearchInput(BaseModel):
    """Input for graph search tool."""
    query: str = Field(..., description="Search query")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")


class DocumentInput(BaseModel):
    """Input for document retrieval."""
    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


class EntityRelationshipInput(BaseModel):
    """Input for entity relationship query."""
    entity_name: str = Field(..., description="Name of the entity")
    depth: int = Field(default=2, description="Maximum traversal depth")


class EntityTimelineInput(BaseModel):
    """Input for entity timeline query."""
    entity_name: str = Field(..., description="Name of the entity")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


# Tool Implementation Functions
async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Perform vector similarity search.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)
        
        # Perform vector search
        results = await vector_search(
            embedding=embedding,
            limit=input_data.limit
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def graph_search_tool(input_data: GraphSearchInput) -> List[GraphSearchResult]:
    """
    Search the knowledge graph.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of graph search results
    """
    try:
        results = await search_knowledge_graph(
            query=input_data.query
        )
        
        # Convert to GraphSearchResult models
        return [
            GraphSearchResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid")
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Perform hybrid search (vector + keyword).
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)
        
        # Perform hybrid search
        results = await hybrid_search(
            embedding=embedding,
            query_text=input_data.query,
            limit=input_data.limit,
            text_weight=input_data.text_weight
        )
        
        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document.
    
    Args:
        input_data: Document retrieval parameters
    
    Returns:
        Document data or None
    """
    try:
        document = await get_document(input_data.document_id)
        
        if document:
            # Also get all chunks for the document
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks
        
        return document
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    List available documents.
    
    Args:
        input_data: Listing parameters
    
    Returns:
        List of document metadata
    """
    try:
        documents = await list_documents(
            limit=input_data.limit,
            offset=input_data.offset
        )
        
        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count")
            )
            for d in documents
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []


async def get_entity_relationships_tool(input_data: EntityRelationshipInput) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Args:
        input_data: Entity relationship parameters
    
    Returns:
        Entity relationships
    """
    try:
        return await get_entity_relationships(
            entity=input_data.entity_name,
            depth=input_data.depth
        )
        
    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        return {
            "central_entity": input_data.entity_name,
            "related_entities": [],
            "relationships": [],
            "depth": input_data.depth,
            "error": str(e)
        }


async def get_entity_timeline_tool(input_data: EntityTimelineInput) -> List[Dict[str, Any]]:
    """
    Get timeline of facts for an entity.
    
    Args:
        input_data: Timeline query parameters
    
    Returns:
        Timeline of facts
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        
        if input_data.start_date:
            start_date = datetime.fromisoformat(input_data.start_date)
        if input_data.end_date:
            end_date = datetime.fromisoformat(input_data.end_date)
        
        # Get timeline from graph
        timeline = await graph_client.get_entity_timeline(
            entity_name=input_data.entity_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return timeline
        
    except Exception as e:
        logger.error(f"Entity timeline query failed: {e}")
        return []


# Combined search function for agent use
async def perform_comprehensive_search(
    query: str,
    use_vector: bool = True,
    use_graph: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform a comprehensive search using multiple methods.
    
    Args:
        query: Search query
        use_vector: Whether to use vector search
        use_graph: Whether to use graph search
        limit: Maximum results per search type (only applies to vector search)
    
    Returns:
        Combined search results
    """
    results = {
        "query": query,
        "vector_results": [],
        "graph_results": [],
        "total_results": 0
    }
    
    tasks = []
    
    if use_vector:
        tasks.append(vector_search_tool(VectorSearchInput(query=query, limit=limit)))
    
    if use_graph:
        tasks.append(graph_search_tool(GraphSearchInput(query=query)))
    
    if tasks:
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if use_vector and not isinstance(search_results[0], Exception):
            results["vector_results"] = search_results[0]
        
        if use_graph:
            graph_idx = 1 if use_vector else 0
            if not isinstance(search_results[graph_idx], Exception):
                results["graph_results"] = search_results[graph_idx]
    
    results["total_results"] = len(results["vector_results"]) + len(results["graph_results"])

    return results


# ---------------------------------------------------------------------------
# PageIndex cloud API retrieval
# ---------------------------------------------------------------------------

class PageIndexSearchInput(BaseModel):
    """Input for PageIndex tree-based retrieval tool."""
    query: str = Field(..., description="Search query to retrieve relevant sections")


def _extract_retrieval_content(retrieval_result: Dict[str, Any]) -> str:
    """
    Extract readable text from a PageIndex get_retrieval() API response.

    Confirmed response schema (observed from live API, 2026-04-02):
        {
            "retrieval_id": "sr-...",
            "doc_id": null | "pi-...",
            "status": "completed",
            "query": "...",
            "retrieved_nodes": [
                {
                    "title":      "Section heading",
                    "node_id":    "0003",
                    "page_index": 1,
                    "text":       "## Section heading\n\nFull section text...",
                    "nodes":      [...]   # optional child nodes
                },
                ...
            ]
        }

    Each item in retrieved_nodes is a tree node with a "text" field containing
    the full markdown section text and a "title" field with the section heading.
    Empty retrieved_nodes means the document has no relevant section for the query.

    Falls back through older field names in case the API schema changes.
    """
    parts: List[str] = []

    # 1. PRIMARY — retrieved_nodes (confirmed schema as of v0.2.8)
    retrieved_nodes = retrieval_result.get("retrieved_nodes")
    if isinstance(retrieved_nodes, list):
        for node in retrieved_nodes:
            if not isinstance(node, dict):
                continue
            title = node.get("title", "")
            text  = node.get("text", "")
            if text and text.strip():
                header = f"[Section: {title}]\n" if title else ""
                parts.append(header + text.strip())
        if parts:
            return "\n\n".join(parts)
        # retrieved_nodes was present but empty — document had no relevant section
        return ""

    # 2. FALLBACK — simple top-level string fields (in case API changes)
    for key in ("answer", "text", "content"):
        val = retrieval_result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # 3. FALLBACK — nested result dict
    result_obj = retrieval_result.get("result")
    if isinstance(result_obj, dict):
        for sub_key in ("text", "content", "answer", "summary"):
            val = result_obj.get(sub_key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    # 4. FALLBACK — other list fields (nodes, sections, data)
    for list_key in ("nodes", "sections", "results", "data"):
        container = retrieval_result.get(list_key)
        if isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("node_id") or ""
                for sub_key in ("text", "content", "summary"):
                    val = item.get(sub_key)
                    if isinstance(val, str) and val.strip():
                        prefix = f"[Section: {title}]\n" if title else ""
                        parts.append(prefix + val.strip())
                        break
            if parts:
                return "\n\n".join(parts)

    return ""


async def pageindex_search_tool(input_data: PageIndexSearchInput) -> str:
    """
    Search documents using the PageIndex cloud API (tree-based retrieval).

    How it works:
      1. Loads the doc_id index from pageindex_trees/index.json
         (built by ingestion/ingest_pageindex.py)
      2. Submits the query to every indexed document in parallel via the
         PageIndex cloud API (submit_query → poll get_retrieval)
      3. Extracts and returns the relevant section text with document labels

    Why parallel queries?  The PageIndex API accepts one doc_id per
    submit_query call.  Running them concurrently keeps latency bounded to
    the slowest single document rather than the sum of all documents.

    Args:
        input_data: PageIndexSearchInput with the query string

    Returns:
        Formatted string with retrieved sections and document names,
        or a helpful error message if no index exists or API key is missing.
    """
    _ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_file = os.path.join(_ROOT_PATH, "pageindex_trees", "index.json")

    # --- Guard: no index yet ---
    if not os.path.exists(index_file):
        return (
            "No PageIndex documents are indexed yet.\n"
            "Run:  python ingestion/ingest_pageindex.py\n"
            "to upload your PDF documents to the PageIndex cloud first."
        )

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception as exc:
        return f"Failed to read PageIndex index: {exc}"

    if not index:
        return (
            "The PageIndex index is empty.\n"
            "Run:  python ingestion/ingest_pageindex.py\n"
            "to upload your PDF documents first."
        )

    # --- Guard: API key ---
    api_key = os.environ.get("PAGEINDEX_API_KEY", "")
    if not api_key:
        return (
            "PAGEINDEX_API_KEY is not set.\n"
            "Add  PAGEINDEX_API_KEY=your_key  to your .env file.\n"
            "Get an API key at https://pageindex.ai"
        )

    try:
        from pageindex import PageIndexClient, PageIndexAPIError
    except ImportError:
        return "PageIndex library not installed. Run: pip install pageindex"

    client = PageIndexClient(api_key=api_key)

    # --- Query each document in parallel ----------------------------------
    async def _query_one(rel_path: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Submit query to one document and poll until done (max 60 s)."""
        doc_id = entry.get("doc_id", "")
        name   = entry.get("name", os.path.basename(rel_path))

        if not doc_id:
            return {"name": name, "error": "missing doc_id in index"}

        try:
            submit_result = await asyncio.to_thread(
                client.submit_query, doc_id, input_data.query
            )
            retrieval_id = submit_result.get("retrieval_id")
            if not retrieval_id:
                return {
                    "name": name,
                    "error": f"no retrieval_id returned: {submit_result}",
                }

            # Poll immediately, then back off.
            # IMPORTANT: the API returns status="completed" within 2-5 seconds
            # for small documents, and the completed task may expire if we wait
            # too long before the first poll (confirmed: 10s sleep caused
            # "Retrieval task not found" errors).
            poll_intervals = [0, 2, 3, 5, 5, 10, 15, 20, 30, 30, 30, 60]
            for gap in poll_intervals:
                await asyncio.sleep(gap)
                try:
                    ret = await asyncio.to_thread(client.get_retrieval, retrieval_id)
                except Exception as exc:
                    logger.warning(f"pageindex get_retrieval error ({name}): {exc}")
                    continue

                status = ret.get("status", "")

                # "completed" is the confirmed terminal status for retrieved_nodes
                is_done = status in ("completed", "done", "finished") or (
                    not status
                    and any(
                        k in ret
                        for k in ("retrieved_nodes", "answer", "text", "content", "result")
                    )
                )
                if is_done:
                    return {"name": name, "result": ret}
                if status == "failed":
                    return {"name": name, "error": "retrieval failed on server"}

            return {"name": name, "error": "retrieval timed out after ~210s"}

        except Exception as exc:
            return {"name": name, "error": str(exc)}

    tasks   = [_query_one(rp, entry) for rp, entry in index.items()]
    results = await asyncio.gather(*tasks)

    # --- Format output ----------------------------------------------------
    parts: List[str] = []
    for r in results:
        name = r.get("name", "Unknown document")
        if "error" in r:
            # Log but don't surface individual errors to the agent —
            # other documents may still have good results.
            logger.warning(f"pageindex_search: {name} — {r['error']}")
            continue

        raw_result = r.get("result", {})
        retrieved_nodes = raw_result.get("retrieved_nodes", [])
        logger.info(
            f"pageindex_search [{name}] — {len(retrieved_nodes)} node(s) returned"
        )
        for node in retrieved_nodes:
            logger.info(
                f"  node_id={node.get('node_id')!r}  "
                f"page={node.get('page_index')}  "
                f"title={node.get('title')!r}  "
                f"text_preview={node.get('text', '')[:200]!r}"
            )

        content = _extract_retrieval_content(raw_result)
        if content:
            parts.append(f"[Document: {name}]\n{content}")

    if not parts:
        return (
            f"PageIndex retrieval returned no results for: '{input_data.query}'.\n"
            "Suggestions:\n"
            "  • Try vector_search or hybrid_search for broader coverage\n"
            "  • Check that ingestion/ingest_pageindex.py completed successfully\n"
            "  • Verify PAGEINDEX_API_KEY is valid"
        )

    return "\n\n---\n\n".join(parts)