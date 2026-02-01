"""FastMCP server for grounded-code-mcp."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from grounded_code_mcp.config import Settings
from grounded_code_mcp.embeddings import EmbeddingClient, get_helpful_error_message
from grounded_code_mcp.manifest import Manifest
from grounded_code_mcp.vectorstore import SearchResult, create_vector_store

logger = logging.getLogger(__name__)

# Create the FastMCP server
mcp = FastMCP("grounded-code-mcp")

# Global state (initialized on startup)
_settings: Settings | None = None
_embedder: EmbeddingClient | None = None
_manifest: Manifest | None = None


def initialize(settings: Settings | None = None) -> None:
    """Initialize the server with settings.

    Args:
        settings: Application settings. If None, loads from default locations.
    """
    global _settings, _embedder, _manifest

    _settings = settings or Settings.load()
    _embedder = EmbeddingClient.from_settings(_settings.ollama)
    _manifest = Manifest.load_or_create(_settings.knowledge_base.manifest_path)


def get_settings() -> Settings:
    """Get the current settings, initializing if needed."""
    if _settings is None:
        initialize()
    if _settings is None:
        raise RuntimeError("Failed to initialize settings")
    return _settings


def get_embedder() -> EmbeddingClient:
    """Get the embedding client, initializing if needed."""
    if _embedder is None:
        initialize()
    if _embedder is None:
        raise RuntimeError("Failed to initialize embedder")
    return _embedder


def get_manifest() -> Manifest:
    """Get the manifest, initializing if needed."""
    if _manifest is None:
        initialize()
    if _manifest is None:
        raise RuntimeError("Failed to initialize manifest")
    return _manifest


def _format_search_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Format search results for tool output.

    Args:
        results: List of SearchResult objects.

    Returns:
        List of formatted result dictionaries.
    """
    return [
        {
            "content": r.content,
            "score": round(r.score, 4),
            "source_path": r.source_path,
            "heading_context": r.heading_context,
            "is_code": r.metadata.get("is_code", False),
            "code_language": r.metadata.get("code_language"),
        }
        for r in results
    ]


# Implementation functions (testable without MCP decorator)


def _search_knowledge_impl(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
    min_score: float = 0.7,
) -> list[dict[str, Any]]:
    """Search the knowledge base for relevant documentation."""
    settings = get_settings()
    embedder = get_embedder()
    store = create_vector_store(settings)

    try:
        embedder.ensure_ready()
    except Exception as e:
        error_msg = get_helpful_error_message(e)
        return [{"error": error_msg}]

    # Generate query embedding
    result = embedder.embed(query)
    query_embedding = result.embedding

    # Determine which collections to search
    if collection:
        collections = [f"{settings.vectorstore.collection_prefix}{collection}"]
    else:
        collections = store.list_collections()

    # Search all target collections
    all_results: list[SearchResult] = []
    for coll in collections:
        try:
            results = store.search(
                coll,
                query_embedding,
                n_results=n_results,
                min_score=min_score,
            )
            all_results.extend(results)
        except Exception as e:
            logger.warning("Error searching collection %s: %s", coll, e)

    # Sort by score and limit
    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:n_results]

    return _format_search_results(all_results)


def _search_code_examples_impl(
    query: str,
    language: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Search for code examples in the knowledge base."""
    settings = get_settings()
    embedder = get_embedder()
    store = create_vector_store(settings)

    try:
        embedder.ensure_ready()
    except Exception as e:
        error_msg = get_helpful_error_message(e)
        return [{"error": error_msg}]

    # Generate query embedding
    result = embedder.embed(query)
    query_embedding = result.embedding

    # Build metadata filter for code blocks
    filter_metadata: dict[str, Any] = {"is_code": True}
    if language:
        filter_metadata["code_language"] = language

    # Search all collections
    collections = store.list_collections()
    all_results: list[SearchResult] = []

    for coll in collections:
        try:
            results = store.search(
                coll,
                query_embedding,
                n_results=n_results,
                min_score=0.5,
                filter_metadata=filter_metadata,
            )
            all_results.extend(results)
        except Exception as e:
            logger.warning("Error searching collection %s: %s", coll, e)

    # Sort by score and limit
    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:n_results]

    # Format for code display
    return [
        {
            "code": r.content,
            "language": r.metadata.get("code_language", "text"),
            "source_path": r.source_path,
            "heading_context": r.heading_context,
            "score": round(r.score, 4),
        }
        for r in all_results
    ]


def _list_collections_impl() -> list[dict[str, Any]]:
    """List all available collections in the knowledge base."""
    settings = get_settings()
    store = create_vector_store(settings)
    manifest = get_manifest()

    collections = store.list_collections()
    result = []

    for coll in collections:
        count = store.collection_count(coll)
        # Get source count from manifest
        sources = manifest.get_sources_by_collection(coll)
        result.append({
            "name": coll,
            "chunk_count": count,
            "source_count": len(sources),
        })

    return result


def _list_sources_impl(collection: str | None = None) -> list[dict[str, Any]]:
    """List all ingested sources in the knowledge base."""
    settings = get_settings()
    manifest = get_manifest()

    if collection:
        full_name = f"{settings.vectorstore.collection_prefix}{collection}"
        sources = manifest.get_sources_by_collection(full_name)
    else:
        sources = list(manifest.sources.values())

    return [
        {
            "path": s.path,
            "collection": s.collection,
            "file_type": s.file_type,
            "title": s.title,
            "chunk_count": s.chunk_count,
            "ingested_at": s.ingested_at.isoformat(),
        }
        for s in sources
    ]


def _get_source_info_impl(source_path: str) -> dict[str, Any]:
    """Get detailed information about a specific source."""
    manifest = get_manifest()
    entry = manifest.get_entry(source_path)

    if not entry:
        return {"error": f"Source not found: {source_path}"}

    return {
        "path": entry.path,
        "collection": entry.collection,
        "file_type": entry.file_type,
        "title": entry.title,
        "page_count": entry.page_count,
        "chunk_count": entry.chunk_count,
        "sha256": entry.sha256,
        "ingested_at": entry.ingested_at.isoformat(),
    }


# MCP tool wrappers


@mcp.tool()
def search_knowledge(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
    min_score: float = 0.7,
) -> list[dict[str, Any]]:
    """Search the knowledge base for relevant documentation.

    Args:
        query: The search query.
        collection: Specific collection to search (optional).
        n_results: Maximum number of results to return.
        min_score: Minimum similarity score (0-1).

    Returns:
        List of matching documents with content, score, and metadata.
    """
    return _search_knowledge_impl(query, collection, n_results, min_score)


@mcp.tool()
def search_code_examples(
    query: str,
    language: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Search for code examples in the knowledge base.

    Args:
        query: The search query (e.g., "async HTTP request").
        language: Filter by programming language (optional).
        n_results: Maximum number of results.

    Returns:
        List of code examples with content, source, and language.
    """
    return _search_code_examples_impl(query, language, n_results)


@mcp.tool()
def list_collections() -> list[dict[str, Any]]:
    """List all available collections in the knowledge base.

    Returns:
        List of collections with name and document count.
    """
    return _list_collections_impl()


@mcp.tool()
def list_sources(collection: str | None = None) -> list[dict[str, Any]]:
    """List all ingested sources in the knowledge base.

    Args:
        collection: Filter by collection name (optional).

    Returns:
        List of sources with path, type, and chunk count.
    """
    return _list_sources_impl(collection)


@mcp.tool()
def get_source_info(source_path: str) -> dict[str, Any]:
    """Get detailed information about a specific source.

    Args:
        source_path: Path to the source file.

    Returns:
        Source information including title, type, chunks, and ingestion date.
    """
    return _get_source_info_impl(source_path)


def run_server(debug: bool = False) -> None:
    """Run the MCP server.

    Args:
        debug: Enable debug mode.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    initialize()
    mcp.run()
