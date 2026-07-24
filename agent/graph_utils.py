"""
Graph utilities for Neo4j/Graphiti integration.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load environment variables BEFORE importing graphiti_core.
# Graphiti 0.12.x routes EVERY query to helpers.DEFAULT_DATABASE, a module-level
# constant = os.getenv('DEFAULT_DATABASE', 'neo4j') evaluated at import time —
# NOT the Graphiti(...).database attribute. Some Neo4j Aura instances name their
# single database after the instance id rather than 'neo4j', so map our
# NEO4J_DATABASE setting onto DEFAULT_DATABASE before the graphiti_core import,
# otherwise every query targets a non-existent 'neo4j' database.
load_dotenv()
_neo4j_db = os.getenv("NEO4J_DATABASE")
if _neo4j_db:
    os.environ["DEFAULT_DATABASE"] = _neo4j_db

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

logger = logging.getLogger(__name__)


def _within_date_range(valid_at: Optional[str], start_d, end_d) -> bool:
    """
    Return True if `valid_at` (an ISO datetime string) falls within the
    inclusive [start_d, end_d] calendar-date range.

    Facts with no parseable `valid_at` are excluded whenever a bound is set,
    because an undated fact cannot be placed inside a date window. Comparison is
    done on calendar dates so timezone-aware graph timestamps compare cleanly
    against naive user-supplied bounds.
    """
    if not valid_at:
        return False
    try:
        d = datetime.fromisoformat(str(valid_at).replace("Z", "+00:00")).date()
    except ValueError:
        return False
    if start_d and d < start_d:
        return False
    if end_d and d > end_d:
        return False
    return True


# Help from this PR for setting up the custom clients: https://github.com/getzep/graphiti/pull/601/files
class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize Graphiti client.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        # LLM configuration
        # GRAPH_LLM_CHOICE overrides LLM_CHOICE for Graphiti's entity-extraction calls.
        # Graphiti requires structured outputs (response_format: json_schema) which not
        # all models support. On Groq use llama-3.1-8b-instant; on OpenAI any model works.
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_choice = os.getenv("GRAPH_LLM_CHOICE") or os.getenv("LLM_CHOICE", "gpt-4.1-mini")
        
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        
        # Embedding configuration
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("VECTOR_DIMENSION", "1536"))
        
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")
        
        self.graphiti: Optional[Graphiti] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Graphiti client."""
        if self._initialized:
            return
        
        try:
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,  # Can be the same as main model
                base_url=self.llm_base_url
            )

            llm_client = OpenAIClient(config=llm_config)

            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url
                )
            )

            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )

            await self.graphiti.build_indices_and_constraints()
            
            self._initialized = True
            logger.info(f"Graphiti client initialized successfully with LLM: {self.llm_choice} and embedder: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise
    
    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")

    def _require_graphiti(self) -> Graphiti:
        """
        Return the initialized Graphiti client, raising if initialize() hasn't run.

        `self.graphiti` is Optional until initialize() constructs it. Routing every
        use through this helper makes the precondition explicit — a clear
        RuntimeError instead of an AttributeError on None later.
        """
        if self.graphiti is None:
            raise RuntimeError(
                "Graphiti client is not initialized; call initialize() first"
            )
        return self.graphiti
    
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an episode to the knowledge graph.
        
        Args:
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata
        """
        if not self._initialized:
            await self.initialize()
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        # Import EpisodeType for proper source handling
        from graphiti_core.nodes import EpisodeType
        
        await self._require_graphiti().add_episode(
            name=episode_id,
            episode_body=content,
            source=EpisodeType.text,  # Always use text type for our content
            source_description=source,
            reference_time=episode_timestamp
        )
        
        logger.info(f"Added episode {episode_id} to knowledge graph")
    
    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search
        
        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's search method (simplified parameters)
            results = await self._require_graphiti().search(query)
            
            # Convert results to dictionaries
            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                    "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
                    "source_node_uuid": str(result.source_node_uuid) if hasattr(result, 'source_node_uuid') and result.source_node_uuid else None
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def get_related_entities(
        self,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get facts related to a given entity using Graphiti semantic search.

        This performs a focused semantic search over the graph's facts for the
        named entity. It does not do multi-hop graph traversal or relationship-type
        filtering — Graphiti's search returns the most relevant facts directly.

        Args:
            entity_name: Name of the entity

        Returns:
            The central entity and the facts semantically related to it
        """
        if not self._initialized:
            await self.initialize()
        
        results = await self._require_graphiti().search(f"relationships involving {entity_name}")
        
        # Extract entity information from the search results
        related_entities = set()
        facts = []
        
        for result in results:
            facts.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None
            })
            
            # Simple entity extraction from fact text (could be enhanced)
            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)
        
        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search"
        }
    
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.

        When start_date and/or end_date are supplied, facts are filtered by the
        calendar date of their `valid_at` timestamp (inclusive). Facts without a
        `valid_at` cannot be placed in time and are dropped whenever any bound is
        set. Date-only comparison is used so timezone-aware graph timestamps and
        naive user-supplied bounds compare safely.

        Args:
            entity_name: Name of the entity
            start_date: Inclusive lower bound (or None for no lower bound)
            end_date: Inclusive upper bound (or None for no upper bound)

        Returns:
            Timeline of facts, newest first
        """
        if not self._initialized:
            await self.initialize()

        results = await self._require_graphiti().search(f"timeline history of {entity_name}")

        timeline = []
        for result in results:
            valid_raw = result.valid_at if hasattr(result, 'valid_at') else None
            timeline.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(valid_raw) if valid_raw else None,
                "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
            })

        # Apply inclusive date-range filtering when bounds are provided.
        if start_date or end_date:
            start_d = start_date.date() if start_date else None
            end_d = end_date.date() if end_date else None
            timeline = [
                item for item in timeline
                if _within_date_range(item.get("valid_at"), start_d, end_d)
            ]

        # Sort by valid_at if available (newest first)
        timeline.sort(key=lambda x: x.get('valid_at') or '', reverse=True)

        return timeline
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, return a simple search to verify the graph is working
        # More detailed statistics would require direct Neo4j access
        try:
            test_results = await self._require_graphiti().search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access"
            }
        except Exception as e:
            return {
                "graphiti_initialized": False,
                "error": str(e)
            }
    
    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's proper clear_data function with the driver
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            # Fallback: Close and reinitialize (this will create fresh indices)
            if self.graphiti:
                await self.graphiti.close()
            
            # Create OpenAI-compatible clients for reinitialization
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,
                base_url=self.llm_base_url
            )
            
            llm_client = OpenAIClient(config=llm_config)
            
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url
                )
            )
            
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )
            await self.graphiti.build_indices_and_constraints()
            
            logger.warning("Reinitialized Graphiti client (fresh indices created)")


# Global Graphiti client instance
graph_client = GraphitiClient()


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


# Convenience functions for common operations
async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add content to the knowledge graph.
    
    Args:
        content: Content to add
        source: Source of the content
        episode_id: Optional episode ID
        metadata: Optional metadata
    
    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    
    await graph_client.add_episode(
        episode_id=episode_id,
        content=content,
        source=source,
        metadata=metadata
    )
    
    return episode_id


async def search_knowledge_graph(
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.
    
    Args:
        query: Search query
    
    Returns:
        Search results
    """
    return await graph_client.search(query)


async def get_entity_relationships(
    entity: str
) -> Dict[str, Any]:
    """
    Get facts related to an entity via Graphiti semantic search.

    Args:
        entity: Entity name

    Returns:
        Entity relationships
    """
    return await graph_client.get_related_entities(entity)


async def test_graph_connection() -> bool:
    """
    Test graph database connection.
    
    Returns:
        True if connection successful
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False


async def test_graph_connection_fast() -> bool:
    """
    Lightweight Neo4j connectivity check for health probes.

    Runs a trivial ``RETURN 1`` against the already-initialized Graphiti driver on
    the configured database. Unlike test_graph_connection() it does NOT embed or
    run a Graphiti search, so it is cheap and makes no LLM/embedding API calls.
    Returns False if the graph client isn't initialized or the query fails.
    """
    try:
        if not graph_client._initialized or graph_client.graphiti is None:
            return False
        database = graph_client.graphiti.database
        async with graph_client.graphiti.driver.session(database=database) as session:
            result = await session.run("RETURN 1 AS ok")
            await result.single()
        return True
    except Exception as e:
        logger.error(f"Graph connectivity check failed: {e}")
        return False