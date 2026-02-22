"""Ingestion pipeline for processing documents into the knowledge base."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from grounded_code_mcp.chunking import DocumentChunker
from grounded_code_mcp.embeddings import EmbeddingClient, get_helpful_error_message
from grounded_code_mcp.manifest import Manifest, SourceEntry, compute_sha256
from grounded_code_mcp.parser import (
    DocumentParseError,
    DocumentParser,
    UnsupportedFormatError,
    get_file_type,
    scan_directory,
)
from grounded_code_mcp.vectorstore import VectorStore, create_vector_store

if TYPE_CHECKING:
    from grounded_code_mcp.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Statistics from an ingestion run."""

    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if no files failed."""
        return self.files_failed == 0


class IngestionPipeline:
    """Pipeline for ingesting documents into the knowledge base."""

    def __init__(
        self,
        settings: Settings,
        *,
        parser: DocumentParser | None = None,
        chunker: DocumentChunker | None = None,
        embedder: EmbeddingClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            settings: Application settings.
            parser: Document parser (optional, created from settings if None).
            chunker: Document chunker (optional, created from settings if None).
            embedder: Embedding client (optional, created from settings if None).
            store: Vector store (optional, created from settings if None).
        """
        self.settings = settings

        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker.from_settings(settings.chunking)
        self.embedder = embedder or EmbeddingClient.from_settings(settings.ollama)
        self.store = store or create_vector_store(settings)

        # Load or create manifest
        self.manifest = Manifest.load_or_create(settings.knowledge_base.manifest_path)

    def ingest(
        self,
        path: Path | None = None,
        *,
        collection: str | None = None,
        force: bool = False,
    ) -> IngestStats:
        """Ingest documents from a path.

        Args:
            path: Path to file or directory. If None, uses sources_dir from settings.
            collection: Target collection name. If None, derived from path.
            force: If True, re-ingest all files regardless of hash.

        Returns:
            IngestStats with results.
        """
        stats = IngestStats()

        # Determine source path
        source_path = path or self.settings.knowledge_base.sources_dir
        if not source_path.exists():
            logger.warning("Source path does not exist: %s", source_path)
            return stats

        # Ensure directories exist
        self.settings.ensure_directories()

        # Check embedder health
        try:
            self.embedder.ensure_ready()
        except Exception as e:
            error_msg = get_helpful_error_message(e)
            logger.error(error_msg)
            stats.errors.append(error_msg)
            stats.files_failed = 1
            return stats

        # Get files to process
        if source_path.is_file():
            files = [source_path]
        else:
            files = scan_directory(source_path, recursive=True)

        stats.files_scanned = len(files)
        logger.info("Found %d files to process", len(files))

        # Process each file
        for file_path in files:
            try:
                status, chunk_count = self._process_file(
                    file_path,
                    collection=collection,
                    force=force,
                )
                if status == "ingested":
                    stats.files_ingested += 1
                    stats.chunks_created += chunk_count
                elif status == "skipped":
                    stats.files_skipped += 1
                elif status == "failed":
                    stats.files_failed += 1
            except Exception as e:
                logger.exception("Failed to process %s", file_path)
                stats.files_failed += 1
                stats.errors.append(f"{file_path}: {e}")

        # Save manifest
        self.manifest.save(self.settings.knowledge_base.manifest_path)

        logger.info(
            "Ingestion complete: %d ingested, %d skipped, %d failed",
            stats.files_ingested,
            stats.files_skipped,
            stats.files_failed,
        )

        return stats

    def _process_file(
        self,
        path: Path,
        *,
        collection: str | None = None,
        force: bool = False,
    ) -> tuple[str, int]:
        """Process a single file.

        Args:
            path: Path to the file.
            collection: Target collection name.
            force: If True, re-ingest regardless of hash.

        Returns:
            Tuple of (status, chunk_count) where status is
            "ingested", "skipped", or "failed".
        """
        relative_path = self._get_relative_path(path)
        collection_name = (
            f"{self.settings.vectorstore.collection_prefix}{collection}"
            if collection
            else self.settings.get_collection_name(path)
        )

        # Check if file needs processing (use relative path for manifest lookup)
        existing_entry = self.manifest.get_entry(str(relative_path))
        if not force and existing_entry is not None:
            current_hash = compute_sha256(path)
            if not existing_entry.has_changed(current_hash):
                logger.debug("Skipping unchanged file: %s", relative_path)
                return ("skipped", 0)

        logger.info("Processing: %s", relative_path)

        # If re-ingesting, delete old chunks first
        if existing_entry and existing_entry.chunk_ids:
            logger.debug("Deleting %d old chunks", len(existing_entry.chunk_ids))
            self.store.delete_chunks(existing_entry.collection, existing_entry.chunk_ids)

        # Parse document
        try:
            parsed = self.parser.parse(path)
        except (UnsupportedFormatError, DocumentParseError) as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return ("failed", 0)

        if parsed.is_empty:
            logger.warning("Empty document: %s", path)
            return ("skipped", 0)

        # Chunk document
        chunks = self.chunker.chunk(parsed.content, source_path=str(relative_path))
        if not chunks:
            logger.warning("No chunks created for: %s", path)
            return ("skipped", 0)

        logger.debug("Created %d chunks", len(chunks))

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embedding_results = self.embedder.embed_many(texts)
        embeddings = [r.embedding for r in embedding_results]

        # Ensure collection exists
        self.store.create_collection(
            collection_name,
            embedding_dim=self.settings.ollama.embedding_dim,
        )

        # Store chunks
        self.store.add_chunks(collection_name, chunks, embeddings)

        # Update manifest
        entry = SourceEntry(
            path=str(relative_path),
            sha256=compute_sha256(path),
            collection=collection_name,
            file_type=get_file_type(path),
            title=parsed.title,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        self.manifest.add_entry(entry)

        return ("ingested", len(chunks))

    def _get_relative_path(self, path: Path) -> Path:
        """Get path relative to sources directory.

        Args:
            path: Absolute path.

        Returns:
            Relative path if within sources_dir, otherwise absolute path.
        """
        try:
            return path.relative_to(self.settings.knowledge_base.sources_dir)
        except ValueError:
            return path

    def remove_source(self, path: Path | str) -> bool:
        """Remove a source and its chunks from the knowledge base.

        Args:
            path: Path to the source file.

        Returns:
            True if source was removed.
        """
        entry = self.manifest.get_entry(path)
        if not entry:
            return False

        # Delete chunks
        if entry.chunk_ids:
            self.store.delete_chunks(entry.collection, entry.chunk_ids)

        # Remove from manifest
        self.manifest.remove_entry(path)
        self.manifest.save(self.settings.knowledge_base.manifest_path)

        return True

    def rebuild_collection(self, collection: str) -> IngestStats:
        """Rebuild an entire collection from scratch.

        Args:
            collection: Collection name to rebuild.

        Returns:
            IngestStats with results.
        """
        # Find all sources in this collection
        sources = self.manifest.get_sources_by_collection(collection)

        # Delete the collection
        self.store.delete_collection(collection)

        # Remove entries from manifest
        for source in sources:
            self.manifest.remove_entry(source.path)

        self.manifest.save(self.settings.knowledge_base.manifest_path)

        # Re-ingest all files
        return self.ingest(force=True)


def ingest_documents(
    settings: Settings,
    path: Path | None = None,
    *,
    collection: str | None = None,
    force: bool = False,
) -> IngestStats:
    """Convenience function to ingest documents.

    Args:
        settings: Application settings.
        path: Path to file or directory.
        collection: Target collection name.
        force: If True, re-ingest all files.

    Returns:
        IngestStats with results.
    """
    pipeline = IngestionPipeline(settings)
    return pipeline.ingest(path, collection=collection, force=force)
