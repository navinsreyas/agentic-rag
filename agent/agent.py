"""
Main Pydantic AI agent for agentic RAG with knowledge graph.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from prometheus_client import Counter
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    pageindex_search_tool,
    generate_embedding,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    PageIndexSearchInput,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Optional[Dict[str, Any]] = None
    # Per-run tool call tracking — both fields are reset to empty/zero on
    # every agent.run() call because AgentDependencies is instantiated fresh
    # each time.  Every @rag_agent.tool function receives ctx.deps pointing
    # to the SAME instance for the lifetime of that single run, so mutations
    # here are visible across all tool calls within the turn.
    tool_call_cache: Dict[str, Any] = field(default_factory=dict)
    tool_call_count: int = 0

    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }


# ---------------------------------------------------------------------------
# Per-turn tool-call guard
# ---------------------------------------------------------------------------

_MAX_TOOL_CALLS_PER_TURN = 2

# Project-specific metric: how often the agent invokes each tool, labeled by tool
# name. Registered on prometheus_client's default REGISTRY, which the FastAPI
# instrumentator's /metrics endpoint exposes. (Generic request/latency metrics
# are handled separately by the instrumentator.)
AGENT_TOOL_INVOCATIONS = Counter(
    "agent_tool_invocations_total",
    "Number of times the agent invoked each tool.",
    ["tool_name"],
)


def _check_tool_limits(
    ctx: RunContext[AgentDependencies],
    cache_key: str,
) -> Optional[Any]:
    """
    Deduplication cache + hard call-count limit, checked at the top of every tool.

    Returns one of three things:
      - The previously stored result (Any)  — same call was already made this turn;
        skip the real API call and return the cached value immediately.
      - A limit-hit string                  — the agent has already called
        _MAX_TOOL_CALLS_PER_TURN tools; tell it to stop and answer.
      - None                                — call is new and within budget;
        the tool should proceed normally.

    Why AgentDependencies?
      deps is created once per agent.run() call and is shared via ctx.deps with
      every tool invocation in that run.  Mutating deps.tool_call_cache and
      deps.tool_call_count here is therefore automatically scoped to a single
      turn — there is no shared state between separate runs.

    Why return a string for the limit case even from list-returning tools?
      Pydantic AI serialises the tool return value to JSON before passing it back
      to the LLM.  A plain string serialises fine regardless of the annotated
      return type, so the model receives an intelligible message rather than an
      empty list that might prompt it to try again.
    """
    # Every @rag_agent.tool passes through here first, so this is the one place
    # to count per-tool invocations. cache_key is always "<tool_name>:<args...>".
    AGENT_TOOL_INVOCATIONS.labels(tool_name=cache_key.split(":", 1)[0]).inc()

    deps = ctx.deps

    if deps.tool_call_count >= _MAX_TOOL_CALLS_PER_TURN:
        logger.warning(
            f"[tool-limit] Hard limit of {_MAX_TOOL_CALLS_PER_TURN} reached "
            f"(cache_key={cache_key!r}, count={deps.tool_call_count}). "
            "Returning limit notice to model."
        )
        return (
            "TOOL CALL LIMIT REACHED: You have already called "
            f"{_MAX_TOOL_CALLS_PER_TURN} tools in this response. "
            "Do not call any more tools. Answer immediately from the "
            "context you have already retrieved."
        )

    if cache_key in deps.tool_call_cache:
        logger.info(
            f"[tool-cache] Duplicate call detected for {cache_key!r}. "
            "Returning cached result without re-calling the API."
        )
        return deps.tool_call_cache[cache_key]

    return None  # proceed with the real tool call


def _store_tool_result(
    ctx: RunContext[AgentDependencies],
    cache_key: str,
    result: Any,
) -> None:
    """
    Increment the call counter and store the result in the per-turn cache.

    Called at the END of each tool, after the real API call completes.
    Must be paired with a successful _check_tool_limits() → None result.
    """
    ctx.deps.tool_call_count += 1
    ctx.deps.tool_call_cache[cache_key] = result
    logger.debug(
        f"[tool-cache] Stored result for {cache_key!r} "
        f"(turn call #{ctx.deps.tool_call_count})"
    )


# Initialize the agent with flexible model configuration
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT
)


# Register tools with proper docstrings (no description parameter)
@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for relevant information using semantic similarity.
    
    This tool performs vector similarity search across document chunks
    to find semantically related content. Returns the most relevant results
    regardless of similarity score.
    
    Args:
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    cache_key = f"vector_search:{query}:{limit}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = VectorSearchInput(
        query=query,
        limit=limit
    )

    results = await vector_search_tool(input_data)

    # Convert results to dict for agent
    output = [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def graph_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for facts and relationships.
    
    This tool queries the knowledge graph to find specific facts, relationships 
    between entities, and temporal information. Best for finding specific facts,
    relationships between companies/people/technologies, and time-based information.
    
    Args:
        query: Search query to find facts and relationships
    
    Returns:
        List of facts with associated episodes and temporal data
    """
    cache_key = f"graph_search:{query}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = GraphSearchInput(query=query)

    results = await graph_search_tool(input_data)

    # Convert results to dict for agent
    output = [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "invalid_at": r.invalid_at,
            "source_node_uuid": r.source_node_uuid
        }
        for r in results
    ]
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform both vector and keyword search for comprehensive results.
    
    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores. Best for combining semantic and exact matching.
    
    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
    
    Returns:
        List of chunks ranked by combined relevance score
    """
    cache_key = f"hybrid_search:{query}:{limit}:{text_weight}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = HybridSearchInput(
        query=query,
        limit=limit,
        text_weight=text_weight
    )

    results = await hybrid_search_tool(input_data)

    # Convert results to dict for agent
    output = [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.
    
    This tool fetches the full document content along with all its chunks
    and metadata. Best for getting comprehensive information from a specific
    source when you need the complete context.
    
    Args:
        document_id: UUID of the document to retrieve
    
    Returns:
        Complete document data with content and metadata, or None if not found
    """
    cache_key = f"get_document:{document_id}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = DocumentInput(document_id=document_id)

    document = await get_document_tool(input_data)

    output = None
    if document:
        output = {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"]
        }
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.
    
    This tool provides an overview of all documents in the knowledge base,
    including titles, sources, and chunk counts. Best for understanding
    what information sources are available.
    
    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination
    
    Returns:
        List of documents with metadata and chunk counts
    """
    cache_key = f"list_documents:{limit}:{offset}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = DocumentListInput(limit=limit, offset=offset)

    documents = await list_documents_tool(input_data)

    output = [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat()
        }
        for d in documents
    ]
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def get_entity_relationships(
    ctx: RunContext[AgentDependencies],
    entity_name: str
) -> Dict[str, Any]:
    """
    Get all relationships for a specific entity in the knowledge graph.

    This tool explores the knowledge graph to find how a specific entity
    (company, person, technology) relates to other entities. Best for
    understanding how concepts, companies, or technologies connect to each other.

    Args:
        entity_name: Name of the entity to explore (e.g., "Google", "OpenAI", "transformer")

    Returns:
        Entity relationships and connected entities
    """
    cache_key = f"get_entity_relationships:{entity_name}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = EntityRelationshipInput(
        entity_name=entity_name
    )

    output = await get_entity_relationships_tool(input_data)
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: str = "",
    end_date: str = ""
) -> List[Dict[str, Any]]:
    """
    Get the timeline of facts for a specific entity.

    This tool retrieves chronological information about an entity,
    showing how information has evolved over time. Best for understanding
    how information about an entity has developed or changed.

    Args:
        entity_name: Name of the entity (e.g., "Microsoft", "AI")
        start_date: Start date in ISO format (YYYY-MM-DD), or empty string for no filter
        end_date: End date in ISO format (YYYY-MM-DD), or empty string for no filter

    Returns:
        Chronological list of facts about the entity with timestamps
    """
    cache_key = f"get_entity_timeline:{entity_name}:{start_date}:{end_date}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = EntityTimelineInput(
        entity_name=entity_name,
        start_date=start_date,
        end_date=end_date
    )

    output = await get_entity_timeline_tool(input_data)
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def page_vector_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> str:
    """
    Retrieve complete pages from documents using semantic similarity. Use when the question asks for full context, complete definitions, or precise wording from a specific section. Best for: 'what does the report say about X', 'explain the full methodology for Y', 'what are the exact steps described for Z'. Returns whole pages preserving complete context unlike chunk-based search.

    Args:
        query: The question or topic to search for in the document pages

    Returns:
        Full text of the most semantically similar pages, with document name,
        page number, and similarity score as headers
    """
    cache_key = f"page_vector_search:{query}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    try:
        import sys
        import os as _os
        _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _pkg_root not in sys.path:
            sys.path.insert(0, _pkg_root)
        from page_index_semantic import semantic_page_search
        output = await semantic_page_search(query, generate_embedding, top_n=3)
    except Exception as exc:
        logger.error(f"page_vector_search error: {exc}")
        output = "Page index search failed."
    _store_tool_result(ctx, cache_key, output)
    return output


@rag_agent.tool
async def pageindex_search(
    ctx: RunContext[AgentDependencies],
    query: str,
) -> str:
    """
    Search documents using the PageIndex cloud API with tree-based LLM reasoning.

    This tool uses VectifyAI's PageIndex service to navigate document structure
    trees and retrieve the most relevant section for the query.  Unlike
    page_vector_search (which uses embedding similarity), this tool uses
    server-side LLM reasoning to traverse the document's hierarchical tree and
    pinpoint the exact section that answers the question.

    Best for:
    - Questions requiring precise section lookup or exact definitions
    - Questions about specific policy wording ("what exactly does X say about Y")
    - Questions with trigger phrases: "exactly", "specifically", "precise wording",
      "what does the policy say", "according to section", "what is the definition of"
    - Structured documents where the answer lives in a named, identifiable section

    NOT the same as page_vector_search, which uses semantic embedding similarity
    to return whole pages.  Use pageindex_search when you need the right section,
    not the right page.

    Args:
        query: The question or information need to retrieve from document trees

    Returns:
        Retrieved section text with document name labels, or a helpful message
        if no documents are indexed or the API key is missing.
    """
    cache_key = f"pageindex_search:{query}"
    early = _check_tool_limits(ctx, cache_key)
    if early is not None:
        return early

    input_data = PageIndexSearchInput(query=query)
    output = await pageindex_search_tool(input_data)
    _store_tool_result(ctx, cache_key, output)
    return output