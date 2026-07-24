"""
FastAPI endpoints for the agentic RAG system.
"""

import os
import time
import asyncio
import json
import logging
from collections import defaultdict, deque
from threading import Lock
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv

from .agent import rag_agent, AgentDependencies
from .providers import get_fallback_model, is_fallback_error, fallback_enabled
from .db_utils import (
    initialize_database,
    close_database,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    test_connection
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection, test_graph_connection_fast
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    SearchType,
    ErrorResponse,
    ToolCall
)
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput
)

load_dotenv()

logger = logging.getLogger(__name__)

# Application configuration
APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
# Cloud Run (and most PaaS) inject their own PORT env var at runtime and expect
# the server to bind it. Prefer PORT; fall back to APP_PORT for local dev, then 8058.
APP_PORT = int(os.getenv("PORT") or os.getenv("APP_PORT") or 8058)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set debug level for our module during development
if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Sentry error tracking (optional, guarded) — initialized before the app is
# created so the ASGI integration wraps request handling from the first request.
# ---------------------------------------------------------------------------
def _init_sentry() -> None:
    """Initialize Sentry only if SENTRY_DSN is set; otherwise skip entirely.

    The app MUST run normally with no DSN (local dev, CI), so a missing/empty
    DSN is a logged no-op, not an error. Any failure during init is swallowed so
    telemetry setup can never take the app down.
    """
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        logger.info("Sentry disabled (SENTRY_DSN not set)")
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.1,
            environment=os.getenv("APP_ENV", "development"),
            # ASGI/FastAPI integration → unhandled exceptions captured automatically.
            integrations=[StarletteIntegration(), FastApiIntegration()],
        )
        logger.info("Sentry enabled (environment=%s)", os.getenv("APP_ENV", "development"))
    except Exception as exc:
        logger.warning(f"Sentry init failed; continuing without it: {exc}")


_init_sentry()


# ---------------------------------------------------------------------------
# Phase 3 guardrails: per-IP rate limiting + input validation
# ---------------------------------------------------------------------------
# All tunable via env; defaults chosen for a low-traffic public demo.
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))

# In-memory per-IP request log. Adequate for a single/low-instance demo, but note
# it is PER-PROCESS: it resets on restart and is NOT shared across multiple Cloud
# Run instances (each instance enforces the limit independently). For a strict
# global limit, back this with Redis or enforce it at an API gateway.
_rate_lock = Lock()
_rate_buckets: Dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Cloud Run / proxies put the real client IP first
    in the X-Forwarded-For header; fall back to the socket peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: fixed-window per-IP limiter. Raises HTTP 429 (with a
    Retry-After header) once an IP exceeds RATE_LIMIT_REQUESTS in the window."""
    ip = _client_ip(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        bucket = _rate_buckets[ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = int(bucket[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests per "
                    f"{RATE_LIMIT_WINDOW_SECONDS // 60} minutes per IP. "
                    f"Try again in ~{retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def validate_query(text: Any, field: str = "query") -> str:
    """Validate a user query: must be a non-empty, non-whitespace string within
    MAX_QUERY_LENGTH. Raises HTTP 400 on failure — before any LLM/DB work."""
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a string.")
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"'{field}' must not be empty or whitespace-only.",
        )
    if len(text) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{field}' exceeds the maximum length of {MAX_QUERY_LENGTH} "
                f"characters (got {len(text)})."
            ),
        )
    return text.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting up agentic RAG API...")
    
    try:
        await initialize_database()
        logger.info("Database initialized")

        await initialize_graph()
        logger.info("Graph database initialized")
        
        # Test connections
        db_ok = await test_connection()
        graph_ok = await test_graph_connection()
        
        if not db_ok:
            logger.error("Database connection failed")
        if not graph_ok:
            logger.error("Graph database connection failed")
        
        logger.info("Agentic RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down agentic RAG API...")
    
    try:
        await close_database()
        await close_graph()
        logger.info("Connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description="AI agent combining vector search and knowledge graph for tech company analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware with flexible CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus: generic request/latency metrics + expose GET /metrics. The custom
# agent_tool_invocations_total counter (defined in agent/agent.py, registered on
# the default registry) is included automatically. /metrics has no rate-limit
# dependency, so scrapers are never throttled.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Helper functions for agent execution
async def get_or_create_session(request: ChatRequest) -> str:
    """Get existing session or create new one."""
    if request.session_id:
        session = await get_session(request.session_id)
        if session:
            return request.session_id

    return await create_session(
        user_id=request.user_id,
        metadata=request.metadata
    )


async def get_conversation_context(
    session_id: str,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Get recent conversation context.
    
    Args:
        session_id: Session ID
        max_messages: Maximum number of messages to retrieve
    
    Returns:
        List of messages
    """
    messages = await get_session_messages(session_id, limit=max_messages)
    
    return [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]


def extract_tool_calls(result) -> List[ToolCall]:
    """
    Extract tool calls from Pydantic AI result.
    
    Args:
        result: Pydantic AI result object
    
    Returns:
        List of ToolCall objects
    """
    tools_used = []
    
    try:
        messages = result.all_messages()
        
        for message in messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Check if this is a tool call part
                    if part.__class__.__name__ == 'ToolCallPart':
                        try:
                            logger.debug(f"ToolCallPart content: tool_name={getattr(part, 'tool_name', None)}")
                            
                            # Extract tool information safely
                            tool_name = str(part.tool_name) if hasattr(part, 'tool_name') else 'unknown'
                            
                            # Get args - the args field is a JSON string in Pydantic AI
                            tool_args = {}
                            if hasattr(part, 'args') and part.args is not None:
                                if isinstance(part.args, str):
                                    # Args is a JSON string, parse it
                                    try:
                                        import json
                                        tool_args = json.loads(part.args)
                                        logger.debug(f"Parsed args from JSON string: {tool_args}")
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Failed to parse args JSON: {e}")
                                        tool_args = {}
                                elif isinstance(part.args, dict):
                                    tool_args = part.args
                                    logger.debug(f"Args already a dict: {tool_args}")
                            
                            # Alternative: use args_as_dict method if available
                            if hasattr(part, 'args_as_dict'):
                                try:
                                    tool_args = part.args_as_dict()
                                    logger.debug(f"Got args from args_as_dict(): {tool_args}")
                                except Exception:
                                    pass
                            
                            tool_call_id = None
                            if hasattr(part, 'tool_call_id'):
                                tool_call_id = str(part.tool_call_id) if part.tool_call_id else None
                            
                            # Create ToolCall with explicit field mapping
                            tool_call_data = {
                                "tool_name": tool_name,
                                "args": tool_args,
                                "tool_call_id": tool_call_id
                            }
                            logger.debug(f"Creating ToolCall with data: {tool_call_data}")
                            tools_used.append(ToolCall(
                                tool_name=tool_name,
                                args=tool_args,
                                tool_call_id=tool_call_id,
                            ))
                        except Exception as e:
                            logger.debug(f"Failed to parse tool call part: {e}")
                            continue
    except Exception as e:
        logger.warning(f"Failed to extract tool calls: {e}")
    
    return tools_used


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a conversation turn to the database.
    
    Args:
        session_id: Session ID
        user_message: User's message
        assistant_message: Assistant's response
        metadata: Optional metadata
    """
    await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=metadata or {}
    )

    await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=metadata or {}
    )


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    save_conversation: bool = True
) -> tuple[str, List[ToolCall]]:
    """
    Execute the agent with a message.
    
    Args:
        message: User message
        session_id: Session ID
        user_id: Optional user ID
        save_conversation: Whether to save the conversation
    
    Returns:
        Tuple of (agent response, tools used)
    """
    try:
        deps = AgentDependencies(
            session_id=session_id,
            user_id=user_id
        )

        context = await get_conversation_context(session_id)

        # Build prompt with context
        full_prompt = message
        if context:
            context_str = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context[-6:]  # Last 3 turns
            ])
            full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {message}"
        
        # Run the agent — try primary model, fall back to OpenAI on tool-call / rate-limit errors
        try:
            result = await rag_agent.run(full_prompt, deps=deps)
        except Exception as primary_exc:
            if fallback_enabled() and is_fallback_error(primary_exc):
                logger.warning(
                    f"Primary model error ({primary_exc!r}), retrying with fallback model"
                )
                fallback = get_fallback_model()
                result = await rag_agent.run(full_prompt, deps=deps, model=fallback)
            else:
                raise

        response = result.data
        tools_used = extract_tool_calls(result)
        
        # Save conversation if requested
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=response,
                metadata={
                    "user_id": user_id,
                    "tool_calls": len(tools_used)
                }
            )
        
        return response, tools_used
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_response = f"I encountered an error while processing your request: {str(e)}"
        
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=error_response,
                metadata={"error": str(e)}
            )
        
        return error_response, []


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check: probes Postgres and Neo4j independently, each with a short
    timeout, so one slow/dead DB can't hang the endpoint."""

    async def _check(coro) -> bool:
        try:
            return bool(await asyncio.wait_for(coro, timeout=3.0))
        except Exception:
            return False

    # Run both probes concurrently so total latency stays ~3s, not ~6s.
    pg_ok, neo_ok = await asyncio.gather(
        _check(test_connection()),
        _check(test_graph_connection_fast()),
    )

    # Always return HTTP 200, even when degraded: Postgres (Neon) and Neo4j (Aura)
    # are EXTERNAL managed services, so restarting this container can't fix a DB
    # blip — a 503 -> Cloud Run restart loop would be pure downside. Report the
    # degraded state in the body for observability instead.
    return {
        "status": "ok" if (pg_ok and neo_ok) else "degraded",
        "postgres": "up" if pg_ok else "down",
        "neo4j": "up" if neo_ok else "down",
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    # Validate input before any LLM/DB work. Kept OUTSIDE the try/except below so
    # the 400 isn't caught and reclassified as a 500.
    validate_query(request.message, "message")
    try:
        session_id = await get_or_create_session(request)

        response, tools_used = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", dependencies=[Depends(rate_limit)])
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    # Validate input before any LLM/DB work (raises 400, outside the try below).
    validate_query(request.message, "message")
    try:
        session_id = await get_or_create_session(request)

        async def generate_stream():
            """Generate streaming response using agent.iter() pattern."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=request.user_id
                )

                context = await get_conversation_context(session_id)

                # Build input with context
                full_prompt = request.message
                if context:
                    context_str = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in context[-6:]
                    ])
                    full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {request.message}"
                
                # Save user message immediately
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={"user_id": request.user_id}
                )
                
                full_response = ""

                # Build a list of models to try: primary (None = agent default),
                # then optional OpenAI fallback if configured.
                from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
                models_to_try = [None]
                if fallback_enabled():
                    models_to_try.append(get_fallback_model())

                run_result = None
                for attempt_idx, model_override in enumerate(models_to_try):
                    is_last_attempt = (attempt_idx == len(models_to_try) - 1)
                    full_response = ""  # reset on each attempt

                    try:
                        iter_kwargs = {"deps": deps}
                        if model_override is not None:
                            iter_kwargs["model"] = model_override

                        async with rag_agent.iter(full_prompt, **iter_kwargs) as run:
                            async for node in run:
                                if rag_agent.is_model_request_node(node):
                                    async with node.stream(run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                                delta_content = event.part.content
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                                full_response += delta_content
                                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                                delta_content = event.delta.content_delta
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                                full_response += delta_content

                        run_result = run.result
                        break  # success — exit retry loop

                    except Exception as stream_exc:
                        if not is_last_attempt and is_fallback_error(stream_exc):
                            logger.warning(
                                f"Primary model stream error ({stream_exc!r}), "
                                "switching to fallback model"
                            )
                            yield f"data: {json.dumps({'type': 'info', 'content': 'Switching to fallback model...'})}\n\n"
                            continue  # retry with next model
                        # Non-retriable error or fallback also failed
                        logger.error(f"Stream error: {stream_exc}")
                        yield f"data: {json.dumps({'type': 'error', 'content': f'Stream error: {str(stream_exc)}'})}\n\n"
                        return

                if run_result is None:
                    return

                tools_used = extract_tool_calls(run_result)
                
                # Send tools used information
                if tools_used:
                    tools_data = [
                        {
                            "tool_name": tool.tool_name,
                            "args": tool.args,
                            "tool_call_id": tool.tool_call_id
                        }
                        for tool in tools_used
                    ]
                    yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\n\n"
                
                await add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={
                        "streamed": True,
                        "tool_calls": len(tools_used)
                    }
                )
                
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                error_chunk = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/vector", dependencies=[Depends(rate_limit)])
async def search_vector(request: SearchRequest):
    """Vector search endpoint."""
    validate_query(request.query, "query")
    try:
        input_data = VectorSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await vector_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type=SearchType.VECTOR,
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/graph", dependencies=[Depends(rate_limit)])
async def search_graph(request: SearchRequest):
    """Knowledge graph search endpoint."""
    validate_query(request.query, "query")
    try:
        input_data = GraphSearchInput(
            query=request.query
        )
        
        start_time = datetime.now()
        results = await graph_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            graph_results=results,
            total_results=len(results),
            search_type=SearchType.GRAPH,
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid", dependencies=[Depends(rate_limit)])
async def search_hybrid(request: SearchRequest):
    """Hybrid search endpoint."""
    validate_query(request.query, "query")
    try:
        input_data = HybridSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await hybrid_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type=SearchType.HYBRID,
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", dependencies=[Depends(rate_limit)])
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0
):
    """List documents endpoint."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler.

    Must return an actual Response — Starlette calls a handler's return value as
    an ASGI app, so returning a raw Pydantic model raised a secondary
    'ErrorResponse object is not callable' TypeError and broke every 500.
    """
    logger.error(f"Unhandled exception: {exc}")

    body = ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4()),
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower()
    )