"""
Document ingestion pipeline: markdown and PDF → PostgreSQL vector DB + knowledge graph.
"""

import os
import logging
import json
import glob
from typing import Callable, List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "PyPDF2 not installed — PDF ingestion will be skipped. "
        "Install it with: pip install PyPDF2"
    )

from dotenv import load_dotenv

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .graph_builder import create_graph_builder

# Import agent utilities
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector DB and knowledge graph."""
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing markdown documents
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        
        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )
        
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing ingestion pipeline...")
        
        # Initialize database connections
        await initialize_database()
        await initialize_graph()
        await self.graph_builder.initialize()
        
        self._initialized = True
        logger.info("Ingestion pipeline initialized")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_database()
            self._initialized = False
    
    async def ingest_documents(
        self,
        progress_callback: Optional[Callable[..., Any]] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents (markdown and PDF) from the documents folder.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()

        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases()

        # Collect all files to process
        markdown_files = self._find_markdown_files()
        pdf_files = self._find_pdf_files()
        all_files = markdown_files + pdf_files

        if not all_files:
            logger.warning(f"No documents found in {self.documents_folder}")
            return []

        logger.info(
            f"Found {len(markdown_files)} markdown file(s) and "
            f"{len(pdf_files)} PDF file(s) to process"
        )

        results = []

        for i, file_path in enumerate(all_files):
            try:
                logger.info(f"Processing file {i+1}/{len(all_files)}: {file_path}")

                if file_path.lower().endswith(".pdf"):
                    result = await self._ingest_pdf_document(file_path)
                else:
                    result = await self._ingest_single_document(file_path)

                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(all_files))

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(
            f"Ingestion complete: {len(results)} documents, "
            f"{total_chunks} chunks, {total_errors} errors"
        )

        return results
    
    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        
        # Read document
        document_content = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        
        # Extract metadata from content
        document_metadata = self._extract_document_metadata(document_content, file_path)
        
        logger.info(f"Processing document: {document_title}")

        # Skip re-ingestion if document already exists in the database
        if await self._document_exists(document_title):
            logger.info(f"Skipping '{document_title}' — already ingested")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=[]
            )

        # Chunk the document
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata
        )
        
        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"]
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Extract entities if configured
        entities_extracted = 0
        if self.config.extract_entities:
            chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
            entities_extracted = sum(
                len(chunk.metadata.get("entities", {}).get("companies", [])) +
                len(chunk.metadata.get("entities", {}).get("technologies", [])) +
                len(chunk.metadata.get("entities", {}).get("people", []))
                for chunk in chunks
            )
            logger.info(f"Extracted {entities_extracted} entities")
        
        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Save to PostgreSQL
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata
        )
        
        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")
        
        # Add to knowledge graph (if enabled)
        relationships_created = 0
        graph_errors = []
        
        if not self.config.skip_graph_building:
            try:
                logger.info("Building knowledge graph relationships (this may take several minutes)...")
                graph_result = await self.graph_builder.add_document_to_graph(
                    chunks=embedded_chunks,
                    document_title=document_title,
                    document_source=document_source,
                    document_metadata=document_metadata
                )
                
                relationships_created = graph_result.get("episodes_created", 0)
                graph_errors = graph_result.get("errors", [])
                
                logger.info(f"Added {relationships_created} episodes to knowledge graph")
                
            except Exception as e:
                error_msg = f"Failed to add to knowledge graph: {str(e)}"
                logger.error(error_msg)
                graph_errors.append(error_msg)
        else:
            logger.info("Skipping knowledge graph building (skip_graph_building=True)")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
            processing_time_ms=processing_time,
            errors=graph_errors
        )
    
    # ------------------------------------------------------------------ PDF support

    def _find_pdf_files(self) -> List[str]:
        """Find all PDF files in the documents folder."""
        if not _PYPDF2_AVAILABLE:
            return []
        if not os.path.exists(self.documents_folder):
            return []
        files = glob.glob(
            os.path.join(self.documents_folder, "**", "*.pdf"), recursive=True
        )
        return sorted(files)

    def _read_pdf_pages(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Read a PDF file and return its pages as (page_number, text) tuples.

        Page numbers are 1-based to match human-readable PDF page references.
        Pages with no extractable text are skipped.
        """
        pages: List[Tuple[int, str]] = []
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append((page_idx + 1, text))
        return pages

    def _extract_pdf_title(self, file_path: str, first_page_text: str) -> str:
        """
        Extract a title for a PDF document.

        Priority:
        1. PDF metadata /Title field
        2. First non-empty short line on page 1 (likely a title heading)
        3. Filename without extension
        """
        if _PYPDF2_AVAILABLE:
            try:
                with open(file_path, "rb") as fh:
                    reader = PyPDF2.PdfReader(fh)
                    info = reader.metadata
                    if info and info.title and info.title.strip():
                        return info.title.strip()
            except Exception:
                pass

        # Try first short line of first page
        for line in first_page_text.split("\n")[:10]:
            line = line.strip()
            if line and len(line) < 120:
                return line

        return os.path.splitext(os.path.basename(file_path))[0]

    async def _document_exists(self, title: str) -> bool:
        """
        Check whether a document with the given title already exists in the DB.

        Returns True if found, False otherwise.
        """
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM documents WHERE title = $1 LIMIT 1",
                title
            )
            return row is not None

    # _save_pages_to_postgres removed — page embedding writes are handled by
    # ingest_page_embeddings() in page_index_semantic.py (Step 3 of semantic PageIndex).

    async def _ingest_pdf_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a PDF document page by page into the vector DB and knowledge graph.

        Each page is chunked separately so that page numbers are preserved
        accurately in chunk metadata.
        """
        start_time = datetime.now()

        if not _PYPDF2_AVAILABLE:
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=0,
                errors=["PyPDF2 not installed — cannot ingest PDF"]
            )

        # Read all pages
        try:
            pages = self._read_pdf_pages(file_path)
        except Exception as e:
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=[f"Failed to read PDF: {e}"]
            )

        if not pages:
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No extractable text found in PDF"]
            )

        first_page_text = pages[0][1]
        document_title = self._extract_pdf_title(file_path, first_page_text)
        document_source = os.path.relpath(file_path, self.documents_folder)
        full_content = "\n\n".join(text for _, text in pages)

        logger.info(f"Processing PDF: {document_title} ({len(pages)} pages)")

        # Skip re-ingestion if document already exists
        if await self._document_exists(document_title):
            logger.info(f"Skipping '{document_title}' — already ingested")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=[]
            )

        document_metadata = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "page_count": len(pages),
            "ingestion_date": datetime.now().isoformat(),
        }

        # Chunk each page separately to preserve page numbers
        all_chunks = []
        for page_number, page_text in pages:
            page_metadata = {
                **document_metadata,
                "page_number": page_number,
            }
            page_chunks = await self.chunker.chunk_document(
                content=page_text,
                title=document_title,
                source=document_source,
                metadata=page_metadata,
            )
            all_chunks.extend(page_chunks)

        # Re-index all chunks globally
        for idx, chunk in enumerate(all_chunks):
            chunk.index = idx

        if not all_chunks:
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created from PDF pages"]
            )

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")

        # Extract entities if configured
        entities_extracted = 0
        if self.config.extract_entities:
            all_chunks = await self.graph_builder.extract_entities_from_chunks(all_chunks)
            entities_extracted = sum(
                len(chunk.metadata.get("entities", {}).get("companies", [])) +
                len(chunk.metadata.get("entities", {}).get("technologies", [])) +
                len(chunk.metadata.get("entities", {}).get("people", []))
                for chunk in all_chunks
            )
            logger.info(f"Extracted {entities_extracted} entities")

        # Generate embeddings for chunks
        embedded_chunks = await self.embedder.embed_chunks(all_chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        # Save document + chunks to PostgreSQL
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            full_content,
            embedded_chunks,
            document_metadata,
        )
        logger.info(f"Saved PDF document to PostgreSQL with ID: {document_id}")

        # Build page-level semantic index: embed each full page as a whole unit
        # and store in page_embeddings for PageIndex semantic retrieval.
        try:
            from page_index_semantic import ingest_page_embeddings
            logger.info(f"Building page-level semantic index for {len(pages)} pages…")
            stored = await ingest_page_embeddings(
                document_id=document_id,
                pages=pages,
                doc_title=document_title,
                source=document_source,
                embedder=self.embedder,
            )
            logger.info(f"Stored {stored} page embeddings in page_embeddings table")
        except Exception as e:
            logger.warning(f"Page-level index failed (non-fatal): {e}")

        # Add to knowledge graph if enabled
        relationships_created = 0
        graph_errors = []

        if not self.config.skip_graph_building:
            try:
                logger.info("Building knowledge graph for PDF (this may take several minutes)...")
                graph_result = await self.graph_builder.add_document_to_graph(
                    chunks=embedded_chunks,
                    document_title=document_title,
                    document_source=document_source,
                    document_metadata=document_metadata,
                )
                relationships_created = graph_result.get("episodes_created", 0)
                graph_errors = graph_result.get("errors", [])
                logger.info(f"Added {relationships_created} episodes to knowledge graph")
            except Exception as e:
                error_msg = f"Failed to add PDF to knowledge graph: {str(e)}"
                logger.error(error_msg)
                graph_errors.append(error_msg)
        else:
            logger.info("Skipping knowledge graph building (skip_graph_building=True)")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(all_chunks),
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
            processing_time_ms=processing_time,
            errors=graph_errors,
        )

    # ------------------------------------------------------------------ markdown support

    def _find_markdown_files(self) -> List[str]:
        """Find all markdown files in the documents folder."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []
        
        patterns = ["*.md", "*.markdown", "*.txt"]
        files = []
        
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.documents_folder, "**", pattern), recursive=True))
        
        return sorted(files)
    
    def _read_document(self, file_path: str) -> str:
        """Read document content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _extract_document_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat()
        }
        
        # Try to extract YAML frontmatter
        if content.startswith('---'):
            try:
                import yaml
                end_marker = content.find('\n---\n', 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning("PyYAML not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")
        
        # Extract some basic metadata from content
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['word_count'] = len(content.split())
        
        return metadata
    
    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata)
                )
                
                document_id = document_result["id"]
                
                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, 'embedding') and chunk.embedding:
                        embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']'
                    
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count
                    )
                
                return document_id
    
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")
        
        # Clean knowledge graph
        await self.graph_builder.clear_graph()
        logger.info("Cleaned knowledge graph")


