"""FastMCP server for grounded-code-mcp."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
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

# Graph RAG constants
MAX_GRAPH_DEPTH = 3
EXPANSION_N_RESULTS_DEFAULT = 5

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
_manifest_mtime: float | None = None
_init_lock = threading.Lock()

# Graph store global state (mtime-based reload)
_graph_store: Any = None  # GraphStore | None
_graph_store_mtime: float | None = None
_graph_lock = threading.Lock()


def _read_manifest_mtime(path: Path) -> float | None:
    """Return the mtime of *path*, or None if the file is missing/unreadable."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def initialize(settings: Settings | None = None) -> None:
    """Initialize the server with settings.

    Args:
        settings: Application settings. If None, loads from default locations.
    """
    global _settings, _embedder, _manifest, _manifest_mtime

    with _init_lock:
        if _settings is not None:
            return
        resolved = settings or Settings.load()
        _embedder = EmbeddingClient.from_settings(resolved.ollama)
        manifest_path = resolved.knowledge_base.manifest_path
        _manifest = Manifest.load_or_create(manifest_path)
        _manifest_mtime = _read_manifest_mtime(manifest_path)
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
    """Get the manifest, reloading from disk if it has changed since last load.

    A CLI ingest run while the server is up writes new entries to
    manifest.json. We detect that via st_mtime so the next MCP tool call
    sees the fresh state without requiring a server restart.
    """
    global _manifest, _manifest_mtime

    if _manifest is None:
        initialize()
    if _manifest is None or _settings is None:
        raise RuntimeError("Failed to initialize manifest")

    manifest_path = _settings.knowledge_base.manifest_path
    current_mtime = _read_manifest_mtime(manifest_path)
    if current_mtime is not None and (_manifest_mtime is None or current_mtime > _manifest_mtime):
        _manifest = Manifest.load_or_create(manifest_path)
        _manifest_mtime = current_mtime

    return _manifest


def _get_graph_store() -> Any:  # -> GraphStore | None
    """Load or reload the concept graph from disk.

    Mirrors the manifest mtime-reload pattern: the CLI may rebuild the graph
    while the server is running; the next tool call picks up the new file
    without requiring a server restart.  Returns None when the graph file is
    absent or unreadable.
    """
    global _graph_store, _graph_store_mtime

    try:
        from graph.graph_store import GraphStore

        env_path = os.environ.get("GRAPH_JSON_PATH")
        if env_path:
            from graph.graph_store import _validate_graph_path

            graph_path = _validate_graph_path(env_path)
        else:
            graph_path = Path("graph") / "concept_graph.json"

        if not graph_path.exists():
            with _graph_lock:
                _graph_store = None
                _graph_store_mtime = None
            return None

        mtime = _read_manifest_mtime(graph_path)
        with _graph_lock:
            if (
                _graph_store is None
                or _graph_store_mtime is None
                or (mtime is not None and mtime > _graph_store_mtime)
            ):
                new_store = GraphStore(path=graph_path)
                new_store.load()
                _graph_store = new_store
                _graph_store_mtime = mtime
        return _graph_store

    except Exception as e:
        logger.warning("Failed to load graph store: %s", e)
        return None


def _extract_concept_ids(query: str, graph: Any) -> list[str]:
    """Match query terms against graph node IDs and descriptions.

    Priority: exact slug match → substring slug match → per-word fallback.
    """
    from graph.graph_store import slugify

    query_slug = slugify(query)

    # 1. Exact slug match
    exact = [m for m in graph.search_nodes(query_slug) if m["id"] == query_slug]
    if exact:
        return [exact[0]["id"]]

    # 2. Substring match on full query slug
    matches = graph.search_nodes(query_slug)
    if matches:
        return [m["id"] for m in matches[:5]]

    # 3. Per-word fallback (skip tokens shorter than 3 chars)
    results: list[str] = []
    seen: set[str] = set()
    for word in query.split():
        word_slug = slugify(word)
        if len(word_slug) < 3:
            continue
        for m in graph.search_nodes(word_slug):
            if m["id"] not in seen:
                seen.add(m["id"])
                results.append(m["id"])

    return results[:10]


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
    expansion_n_results: int = EXPANSION_N_RESULTS_DEFAULT,
) -> list[dict[str, Any]]:
    """Search the knowledge base with optional graph expansion.

    Returns direct vector hits (labeled [vector]) first, followed by
    graph-expanded hits (labeled [graph-expanded: via <concept>]).
    Falls back to pure vector results when the graph is empty or unavailable.
    """
    settings = get_settings()

    # M2: cap query length before sending to the embedding model
    query = query[:MAX_QUERY_CHARS]

    # M1: clamp numeric parameters to safe ranges
    n_results = max(MIN_N_RESULTS, min(n_results, MAX_N_RESULTS))
    min_score = max(MIN_SCORE, min(min_score, MAX_SCORE))

    # M3: validate collection against the configured allowlist.
    # Skip when collections={} — config not found from the caller's cwd.
    if collection is not None:
        known_collections = set(settings.collections.values())
        if known_collections and collection not in known_collections:
            return [{"error": f"Unknown collection: {collection!r}"}]

    embedder = get_embedder()
    vstore = create_vector_store(settings)

    try:
        embedder.ensure_ready()
    except Exception as e:
        error_msg = get_helpful_error_message(e)
        return [{"error": error_msg}]

    # Generate query embedding
    embed_result = embedder.embed(query, is_query=True)
    query_embedding = embed_result.embedding

    # Determine which collections to search
    if collection:
        target_collections = [f"{settings.vectorstore.collection_prefix}{collection}"]
    else:
        target_collections = vstore.list_collections()

    # --- Step 1: Vector search ---
    vector_results: list[SearchResult] = []
    for coll in target_collections:
        try:
            hits = vstore.search(coll, query_embedding, n_results=n_results, min_score=min_score)
            vector_results.extend(hits)
        except Exception as e:
            logger.warning("Error searching collection %s: %s", coll, e)

    vector_results.sort(key=lambda x: x.score, reverse=True)
    vector_results = vector_results[:n_results]
    seen_chunk_ids: set[str] = {r.chunk_id for r in vector_results}

    # --- Steps 2-5: Graph expansion (gracefully skipped when graph is unavailable) ---
    expansion_hits: list[tuple[SearchResult, str]] = []  # (result, via_concept_id)
    try:
        graph = _get_graph_store()
        if graph is not None and graph.node_count > 0:
            concept_ids = _extract_concept_ids(query, graph)
            if concept_ids:
                # Collect source_slugs reachable from matched concepts (depth=2)
                expanded_slugs: dict[str, str] = {}  # source_slug -> via concept id
                for concept_id in concept_ids:
                    for neighbor in graph.get_neighbors(concept_id, depth=2):
                        slug = neighbor.get("source_slug", "")
                        if slug and slug not in expanded_slugs:
                            expanded_slugs[slug] = concept_id

                if expanded_slugs:
                    # Broader second search; filter by source_path prefix in Python
                    exp_limit = (n_results + expansion_n_results) * 3
                    for coll in target_collections:
                        try:
                            raw = vstore.search(
                                coll,
                                query_embedding,
                                n_results=exp_limit,
                                min_score=min_score,
                            )
                            for hit in raw:
                                if hit.chunk_id in seen_chunk_ids:
                                    continue
                                source_path = hit.metadata.get("source_path", "")
                                for slug, via in expanded_slugs.items():
                                    if source_path == slug or source_path.startswith(slug + "/"):
                                        expansion_hits.append((hit, via))
                                        seen_chunk_ids.add(hit.chunk_id)
                                        break
                        except Exception as e:
                            logger.warning("Graph expansion search error for %s: %s", coll, e)

                    expansion_hits.sort(key=lambda t: t[0].score, reverse=True)
                    expansion_hits = expansion_hits[:expansion_n_results]
    except Exception as e:
        logger.warning("Graph expansion failed: %s", e)

    # --- Format and merge: vector hits first, then graph-expanded ---
    formatted: list[dict[str, Any]] = [
        {**_format_search_results([r])[0], "retrieval_type": "[vector]"} for r in vector_results
    ]
    formatted.extend(
        {**_format_search_results([r])[0], "retrieval_type": f"[graph-expanded: via {via}]"}
        for r, via in expansion_hits
    )
    return formatted


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


def _query_graph_impl(
    concept: str,
    depth: int = 2,
    domain: str | None = None,
) -> dict[str, Any]:
    """Query the concept graph and return structured neighborhood information."""
    depth = max(1, min(depth, MAX_GRAPH_DEPTH))
    concept = concept[:MAX_QUERY_CHARS]

    graph = _get_graph_store()
    if graph is None or graph.node_count == 0:
        return {
            "matched_nodes": [],
            "relationships": [],
            "linked_sources": [],
            "summary": "No concept graph is available.",
        }

    concept_ids = _extract_concept_ids(concept, graph)
    if not concept_ids:
        return {
            "matched_nodes": [],
            "relationships": [],
            "linked_sources": [],
            "summary": f"No concepts matching '{concept[:200]}' found in the graph.",
        }

    primary_id = concept_ids[0]

    # Traverse all matched concept IDs and merge their neighborhoods
    all_nodes_dict: dict[str, dict[str, Any]] = {}
    for cid in concept_ids:
        candidates = [m for m in graph.search_nodes(cid) if m["id"] == cid]
        root_node: dict[str, Any] = candidates[0] if candidates else {"id": cid}
        all_nodes_dict[cid] = root_node
        for neighbor in graph.get_neighbors(cid, depth=depth):
            all_nodes_dict.setdefault(neighbor["id"], neighbor)
    all_nodes: list[dict[str, Any]] = list(all_nodes_dict.values())

    # Keep primary_id for backward-compatible summary language
    primary_node: dict[str, Any] = all_nodes_dict.get(primary_id, {"id": primary_id})

    # Optional domain filter
    if domain:
        all_nodes = [n for n in all_nodes if n.get("domain", "") == domain]

    all_node_ids = {n["id"] for n in all_nodes}
    relationships = graph.get_edges_for_nodes(all_node_ids)
    linked_sources = sorted({n.get("source_slug", "") for n in all_nodes if n.get("source_slug")})

    # Inline summary — no LLM call
    node_type = primary_node.get("type", "concept") or "concept"
    node_domain = primary_node.get("domain", "")
    neighbor_count = len(all_nodes) - 1

    if len(concept_ids) > 1:
        parts = [
            f"Graph traversal from {len(concept_ids)} matched concept(s): "
            f"{', '.join(concept_ids[:3])}{'...' if len(concept_ids) > 3 else ''}"
        ]
    else:
        parts = [f"The graph describes '{primary_id}' as a {node_type}"]
    if node_domain:
        parts.append(f"in the {node_domain} domain")
    if neighbor_count > 0:
        parts.append(f"with {neighbor_count} related concept(s) within {depth} hop(s)")
    if linked_sources:
        parts.append(f"sourced from: {', '.join(linked_sources)}")
    summary = " ".join(parts) + "."

    if relationships:
        rel_snippets = [f"{r['from']} {r['rel']} {r['to']}" for r in relationships[:3]]
        summary += f" Key relationships: {'; '.join(rel_snippets)}."

    return {
        "matched_concept_ids": concept_ids,
        "matched_nodes": [
            {
                "id": n["id"],
                "type": n.get("type", ""),
                "domain": n.get("domain", ""),
                "description": n.get("description", ""),
                "source_slug": n.get("source_slug", ""),
            }
            for n in all_nodes
        ],
        "relationships": relationships,
        "linked_sources": linked_sources,
        "summary": summary,
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


@mcp.tool()
def query_graph(
    concept: str,
    depth: int = 2,
    domain: str | None = None,
) -> dict[str, Any]:
    """Query the concept graph for a concept and its neighborhood.

    Args:
        concept: Concept name or search term to look up in the graph.
        depth: Traversal depth from each matched concept node (default 2, max 3). All concepts matching the query are used as traversal roots.
        domain: Optional domain filter (e.g. "architecture", "testing").

    Returns:
        matched_nodes: Nodes matching the concept and their neighbors.
        relationships: Edges (triples) between the matched nodes.
        linked_sources: Source slugs reachable from matched nodes.
        summary: Plain-English description of the concept's graph neighborhood.
    """
    return _query_graph_impl(concept, depth, domain)


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
