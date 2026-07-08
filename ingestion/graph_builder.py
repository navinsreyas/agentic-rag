"""
Knowledge graph builder for extracting entities and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import asyncio
import re

from graphiti_core import Graphiti
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    
    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3  # Reduced batch size for Graphiti
    ) -> Dict[str, Any]:
        """
        Add document chunks to the knowledge graph.
        
        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata
            batch_size: Number of chunks to process in each batch
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"episodes_created": 0, "errors": []}
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")
        logger.info("⚠️ Large chunks will be truncated to avoid Graphiti token limits.")
        
        # Check for oversized chunks and warn
        oversized_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.content) > 6000]
        if oversized_chunks:
            logger.warning(f"Found {len(oversized_chunks)} chunks over 6000 chars that will be truncated: {oversized_chunks}")
        
        episodes_created = 0
        errors = []
        
        # Process chunks one by one to avoid overwhelming Graphiti
        for i, chunk in enumerate(chunks):
            try:
                # Build a human-readable episode ID that includes document title
                # and page number so graph facts can be traced back to source pages.
                page_number = chunk.metadata.get("page_number", "")
                page_suffix = f"_p{page_number}" if page_number else ""
                safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', document_title[:40])
                episode_id = f"{safe_title}{page_suffix}_chunk{chunk.index}"

                # Prepare episode content with size limits
                episode_content = self._prepare_episode_content(
                    chunk,
                    document_title,
                    document_metadata
                )

                # Source description includes page number for traceability
                if page_number:
                    source_description = (
                        f"Document: {document_title} | Page: {page_number} | Chunk: {chunk.index}"
                    )
                else:
                    source_description = f"Document: {document_title} (Chunk: {chunk.index})"
                
                # Add episode to graph
                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": document_title,
                        "document_source": document_source,
                        "page_number": page_number,
                        "chunk_index": chunk.index,
                        "original_length": len(chunk.content),
                        "processed_length": len(episode_content)
                    }
                )
                
                episodes_created += 1
                logger.info(f"✓ Added episode {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})")
                
                # Small delay between each episode to reduce API pressure
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Continue processing other chunks even if one fails
                continue
        
        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Graph building complete: {episodes_created} episodes created, {len(errors)} errors")
        return result
    
    def _prepare_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare episode content with minimal context to avoid token limits.
        
        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata
        
        Returns:
            Formatted episode content (optimized for Graphiti)
        """
        # Limit chunk content to avoid Graphiti's 8192 token limit
        # Estimate ~4 chars per token, keep content under 6000 chars to leave room for processing
        max_content_length = 6000
        
        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )
            
            if last_sentence_end > max_content_length * 0.7:  # If we can keep 70% and end cleanly
                content = truncated[:last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"
            
            logger.warning(f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars for Graphiti")
        
        # Add minimal context (just document title for now)
        if document_title and len(content) < max_content_length - 100:
            episode_content = f"[Doc: {document_title[:50]}]\n\n{content}"
        else:
            episode_content = content
        
        return episode_content
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)."""
        return len(text) // 4
    
    def _is_content_too_large(self, content: str, max_tokens: int = 7000) -> bool:
        """Check if content is too large for Graphiti processing."""
        return self._estimate_tokens(content) > max_tokens
    
    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_companies: bool = True,
        extract_technologies: bool = True,
        extract_people: bool = True
    ) -> List[DocumentChunk]:
        """
        Extract entities from chunks and add to metadata.
        
        Args:
            chunks: List of document chunks
            extract_companies: Whether to extract company names
            extract_technologies: Whether to extract technology terms
            extract_people: Whether to extract person names
        
        Returns:
            Chunks with entity metadata added
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")
        
        enriched_chunks = []
        
        for chunk in chunks:
            entities = {
                "companies": [],
                "technologies": [],
                "people": [],
                "locations": []
            }
            
            content = chunk.content
            
            # Extract companies
            if extract_companies:
                entities["companies"] = self._extract_companies(content)
            
            # Extract technologies
            if extract_technologies:
                entities["technologies"] = self._extract_technologies(content)
            
            # Extract people
            if extract_people:
                entities["people"] = self._extract_people(content)
            
            # Extract locations
            entities["locations"] = self._extract_locations(content)
            
            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entities": entities,
                    "entity_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )
            
            # Preserve embedding if it exists
            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info("Entity extraction complete")
        return enriched_chunks
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract company names from text."""
        # Known tech companies (extend this list as needed)
        tech_companies = {
            "Google", "Microsoft", "Apple", "Amazon", "Meta", "Facebook",
            "Tesla", "OpenAI", "Anthropic", "Nvidia", "Intel", "AMD",
            "IBM", "Oracle", "Salesforce", "Adobe", "Netflix", "Uber",
            "Airbnb", "Spotify", "Twitter", "LinkedIn", "Snapchat",
            "TikTok", "ByteDance", "Baidu", "Alibaba", "Tencent",
            "Samsung", "Sony", "Huawei", "Xiaomi", "DeepMind"
        }
        
        found_companies = set()
        text_lower = text.lower()
        
        for company in tech_companies:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(company.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_companies.add(company)
        
        return list(found_companies)
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technology terms from text."""
        tech_terms = {
            "AI", "artificial intelligence", "machine learning", "ML",
            "deep learning", "neural network", "LLM", "large language model",
            "GPT", "transformer", "NLP", "natural language processing",
            "computer vision", "reinforcement learning", "generative AI",
            "foundation model", "multimodal", "chatbot", "API",
            "cloud computing", "edge computing", "quantum computing",
            "blockchain", "cryptocurrency", "IoT", "5G", "AR", "VR",
            "autonomous vehicles", "robotics", "automation"
        }
        
        found_terms = set()
        text_lower = text.lower()
        
        for term in tech_terms:
            if term.lower() in text_lower:
                found_terms.add(term)
        
        return list(found_terms)
    
    def _extract_people(self, text: str) -> List[str]:
        """Extract person names from text."""
        # Known tech leaders (extend this list as needed)
        tech_leaders = {
            "Elon Musk", "Jeff Bezos", "Tim Cook", "Satya Nadella",
            "Sundar Pichai", "Mark Zuckerberg", "Sam Altman",
            "Dario Amodei", "Daniela Amodei", "Jensen Huang",
            "Bill Gates", "Larry Page", "Sergey Brin", "Jack Dorsey",
            "Reed Hastings", "Marc Benioff", "Andy Jassy"
        }
        
        found_people = set()
        
        for person in tech_leaders:
            if person in text:
                found_people.add(person)
        
        return list(found_people)
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        locations = {
            "Silicon Valley", "San Francisco", "Seattle", "Austin",
            "New York", "Boston", "London", "Tel Aviv", "Singapore",
            "Beijing", "Shanghai", "Tokyo", "Seoul", "Bangalore",
            "Mountain View", "Cupertino", "Redmond", "Menlo Park"
        }
        
        found_locations = set()
        
        for location in locations:
            if location in text:
                found_locations.add(location)
        
        return list(found_locations)
    
    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()