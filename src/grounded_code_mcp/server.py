"""FastMCP server for grounded-code-mcp."""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

from fastmcp import FastMCP

from grounded_code_mcp.config import Settings
from grounded_code_mcp.embeddings import EmbeddingClient, get_helpful_error_message
from grounded_code_mcp.manifest import Manifest
from grounded_code_mcp.vectorstore import SearchResult, create_vector_store

logger = logging.getLogger(__name__)

# Input validation constants
MAX_QUERY_CHARS = 4_000  # ~1 000 tokens; well inside the 8 192-token model limit
MAX_N_RESULTS = 50
MIN_N_RESULTS = 1
MIN_SCORE = 0.0
MAX_SCORE = 1.0
MAX_SOURCE_PATH_CHARS = 256

# Recognised programming languages for the code-search filter (L3).
# Unknown values are silently ignored rather than forwarded to the store.
KNOWN_LANGUAGES: frozenset[str] = frozenset(
    {
        "python",
        "csharp",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "cpp",
        "c",
        "sql",
        "bash",
        "shell",
        "yaml",
        "json",
        "html",
        "css",
        "markdown",
        "php",
        "ruby",
        "scala",
        "kotlin",
        "swift",
        "r",
        "text",
    }
)

# Create the FastMCP server
mcp = FastMCP("grounded-code-mcp")

# Global state (initialized on startup)
_settings: Settings | None = None
_embedder: EmbeddingClient | None = None
_manifest: Manifest | None = None
_init_lock = threading.Lock()


def initialize(settings: Settings | None = None) -> None:
    """Initialize the server with settings.

    Args:
        settings: Application settings. If None, loads from default locations.
    """
    global _settings, _embedder, _manifest

    with _init_lock:
        if _settings is not None:
            return
        resolved = settings or Settings.load()
        _embedder = EmbeddingClient.from_settings(resolved.ollama)
        _manifest = Manifest.load_or_create(resolved.knowledge_base.manifest_path)
        # Assign last so other threads see a fully initialised state
        _settings = resolved


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
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Search the knowledge base for relevant documentation."""
    settings = get_settings()

    # M2: cap query length before sending to the embedding model
    query = query[:MAX_QUERY_CHARS]

    # M1: clamp numeric parameters to safe ranges
    n_results = max(MIN_N_RESULTS, min(n_results, MAX_N_RESULTS))
    min_score = max(MIN_SCORE, min(min_score, MAX_SCORE))

    # M3: validate collection against the configured allowlist
    if collection is not None:
        known_collections = set(settings.collections.values())
        if collection not in known_collections:
            return [{"error": f"Unknown collection: {collection!r}"}]

    embedder = get_embedder()
    store = create_vector_store(settings)

    try:
        embedder.ensure_ready()
    except Exception as e:
        error_msg = get_helpful_error_message(e)
        return [{"error": error_msg}]

    # Generate query embedding
    result = embedder.embed(query, is_query=True)
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

    # M2: cap query length before sending to the embedding model
    query = query[:MAX_QUERY_CHARS]

    # M1: clamp n_results to safe range
    n_results = max(MIN_N_RESULTS, min(n_results, MAX_N_RESULTS))

    embedder = get_embedder()
    store = create_vector_store(settings)

    try:
        embedder.ensure_ready()
    except Exception as e:
        error_msg = get_helpful_error_message(e)
        return [{"error": error_msg}]

    # Generate query embedding
    result = embedder.embed(query, is_query=True)
    query_embedding = result.embedding

    # Build metadata filter for code blocks.
    # L3: only forward recognised language values; unknown strings are ignored
    # rather than passed to the store, preventing crafted filter injection.
    filter_metadata: dict[str, Any] = {"is_code": True}
    if language and language in KNOWN_LANGUAGES:
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
                min_score=0.3,
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
        result.append(
            {
                "name": coll,
                "chunk_count": count,
                "source_count": len(sources),
            }
        )

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
    # L2: cap length before lookup; do not echo raw input in the error response.
    source_path = source_path[:MAX_SOURCE_PATH_CHARS]

    manifest = get_manifest()
    entry = manifest.get_entry(source_path)

    if not entry:
        return {"error": "Source not found"}

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
    min_score: float = 0.3,
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


def run_server(
    debug: bool = False,
    transport: Literal["stdio", "sse", "streamable-http"] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the MCP server.

    Args:
        debug: Enable debug mode.
        transport: Transport protocol ("stdio", "sse", "streamable-http").
            Defaults to None (stdio).
        host: Host to bind HTTP transport. Defaults to "127.0.0.1".
        port: Port for HTTP transport. Defaults to 8080.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    initialize()

    if transport:
        # L4: warn when binding HTTP transport to a non-loopback address.
        # This server has no authentication; exposing it beyond localhost is
        # a deliberate opt-in that operators must acknowledge (OWASP MCP §1).
        if host != "127.0.0.1":
            logger.warning(
                "HTTP transport bound to %s — this server has no authentication. "
                "Use 127.0.0.1 (loopback) unless you have added your own auth layer.",
                host,
            )
        mcp.run(transport=transport, host=host, port=port)
    else:
        mcp.run()
