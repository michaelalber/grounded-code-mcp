"""Tests for graph RAG: hybrid search and query_graph MCP tool.

Coverage:
- Hybrid query: vector-only fallback, graph expansion, deduplication
- query_graph tool: known concept, unknown concept, domain filter, depth limit
- Integration: real NetworkX graph with mocked Qdrant
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graph.graph_store import GraphStore
from grounded_code_mcp.config import (
    KnowledgeBaseSettings,
    OllamaSettings,
    Settings,
    VectorStoreSettings,
)
from grounded_code_mcp.embeddings import EmbeddingResult
from grounded_code_mcp.server import (
    _extract_concept_ids,
    _query_graph_impl,
    _search_knowledge_impl,
)
from grounded_code_mcp.vectorstore import SearchResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(temp_dir: Path) -> Settings:
    sources_dir = temp_dir / "sources"
    sources_dir.mkdir()
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    return Settings(
        knowledge_base=KnowledgeBaseSettings(sources_dir=sources_dir, data_dir=data_dir),
        ollama=OllamaSettings(model="test-model", embedding_dim=128),
        vectorstore=VectorStoreSettings(provider="qdrant", collection_prefix="grounded_"),
        collections={"sources/python": "python"},
    )


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.ensure_ready.return_value = None
    embedder.embed.return_value = EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
    return embedder


def _make_search_result(
    chunk_id: str,
    source_path: str,
    score: float = 0.9,
    content: str = "Test content",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={"source_path": source_path, "heading_context": [], "is_code": False},
    )


def _make_graph_store(
    temp_dir: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> GraphStore:
    """Build a real GraphStore with the given nodes and edges."""
    store = GraphStore(path=temp_dir / "graph.json")
    store.load()
    store.merge_nodes(nodes, edges)
    store.save()
    return store


def _make_node(
    concept_id: str,
    source_slug: str = "my-source",
    domain: str = "architecture",
    node_type: str = "pattern",
    description: str = "",
) -> dict[str, Any]:
    return {
        "id": concept_id,
        "source_slug": source_slug,
        "domain": domain,
        "type": node_type,
        "description": description,
    }


def _make_edge(source: str, target: str, rel: str = "enables") -> dict[str, Any]:
    return {"source": source, "target": target, "rel": rel}


# ---------------------------------------------------------------------------
# _extract_concept_ids
# ---------------------------------------------------------------------------


class TestExtractConceptIds:
    def test_exact_slug_match_returned_first(self, temp_dir: Path) -> None:
        store = _make_graph_store(
            temp_dir,
            [_make_node("vertical-slice"), _make_node("vertical-alignment")],
            [],
        )
        result = _extract_concept_ids("Vertical Slice", store)
        assert result[0] == "vertical-slice"

    def test_substring_match_when_no_exact(self, temp_dir: Path) -> None:
        store = _make_graph_store(temp_dir, [_make_node("vertical-slice")], [])
        result = _extract_concept_ids("vertical", store)
        assert "vertical-slice" in result

    def test_per_word_fallback_when_no_slug_match(self, temp_dir: Path) -> None:
        store = _make_graph_store(temp_dir, [_make_node("ownership")], [])
        result = _extract_concept_ids("rust ownership model", store)
        assert "ownership" in result

    def test_short_words_skipped_in_fallback(self, temp_dir: Path) -> None:
        store = _make_graph_store(temp_dir, [_make_node("on")], [])
        # "on" is 2 chars — below the 3-char threshold
        result = _extract_concept_ids("turn on the system", store)
        assert "on" not in result

    def test_empty_graph_returns_empty_list(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "empty.json")
        store.load()
        assert _extract_concept_ids("anything", store) == []

    def test_unmatched_query_returns_empty_list(self, temp_dir: Path) -> None:
        store = _make_graph_store(temp_dir, [_make_node("cqrs")], [])
        assert _extract_concept_ids("zzznomatch", store) == []

    def test_returns_at_most_ten_results(self, temp_dir: Path) -> None:
        nodes = [_make_node(f"alpha-{i}") for i in range(20)]
        store = _make_graph_store(temp_dir, nodes, [])
        result = _extract_concept_ids("alpha", store)
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# Hybrid query — _search_knowledge_impl
# ---------------------------------------------------------------------------


class TestHybridQueryVectorOnly:
    """When the graph has no matching concepts, return pure vector results."""

    def test_vector_results_returned_when_graph_is_none(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [_make_search_result("chunk-1", "python/doc.md")]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=None),
        ):
            results = _search_knowledge_impl("test query")

        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["retrieval_type"] == "[vector]"

    def test_vector_results_returned_when_graph_is_empty(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [_make_search_result("chunk-1", "python/doc.md")]

        empty_store = GraphStore(path=temp_dir / "empty.json")
        empty_store.load()

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=empty_store),
        ):
            results = _search_knowledge_impl("test query")

        # No expansion search should be triggered for empty graph
        assert mock_store.search.call_count == 1
        assert results[0]["retrieval_type"] == "[vector]"

    def test_no_expansion_when_query_matches_no_concepts(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [_make_search_result("chunk-1", "python/doc.md")]

        graph = _make_graph_store(temp_dir, [_make_node("cqrs")], [])

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("zzznomatch query")

        assert mock_store.search.call_count == 1
        assert all(r["retrieval_type"] == "[vector]" for r in results)


class TestHybridQueryGraphExpansion:
    """Graph expansion adds labeled results when concepts match."""

    def test_expansion_results_labeled_correctly(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        """Graph-expanded results carry the [graph-expanded: via <concept>] label."""
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="patterns-book"),
            ],
            [_make_edge("vertical-slice", "cqrs")],
        )

        vector_hit = _make_search_result("v-1", "python/doc.md", score=0.95)
        expansion_hit = _make_search_result("e-1", "patterns-book/ch1.md", score=0.7)

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.side_effect = [
            [vector_hit],  # first call: vector search
            [vector_hit, expansion_hit],  # second call: expansion search (broader)
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("cqrs pattern")

        vector_results = [r for r in results if r["retrieval_type"] == "[vector]"]
        graph_results = [r for r in results if "[graph-expanded" in r["retrieval_type"]]
        assert len(vector_results) == 1
        assert len(graph_results) == 1
        assert "cqrs" in graph_results[0]["retrieval_type"]

    def test_expansion_only_includes_matching_source_slugs(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        """Expansion hits whose source_path doesn't match any slug are excluded."""
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="patterns-book"),
            ],
            [_make_edge("vertical-slice", "cqrs")],
        )

        vector_hit = _make_search_result("v-1", "python/doc.md", score=0.95)
        # This hit's source path does NOT start with "patterns-book/"
        non_matching_hit = _make_search_result("nm-1", "unrelated/doc.md", score=0.8)

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.side_effect = [
            [vector_hit],
            [vector_hit, non_matching_hit],
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("cqrs pattern")

        graph_results = [r for r in results if "[graph-expanded" in r["retrieval_type"]]
        assert len(graph_results) == 0

    def test_vector_results_appear_before_expansion_results(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="patterns-book"),
            ],
            [_make_edge("vertical-slice", "cqrs")],
        )

        vector_hit = _make_search_result("v-1", "python/doc.md", score=0.95)
        expansion_hit = _make_search_result("e-1", "patterns-book/ch1.md", score=0.7)

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.side_effect = [
            [vector_hit],
            [vector_hit, expansion_hit],
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("cqrs pattern")

        assert results[0]["retrieval_type"] == "[vector]"
        assert "[graph-expanded" in results[-1]["retrieval_type"]


class TestHybridQueryDeduplication:
    """Chunks already in vector results are not duplicated in expansion."""

    def test_chunk_in_both_vector_and_expansion_appears_once(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="patterns-book"),
            ],
            [_make_edge("vertical-slice", "cqrs")],
        )

        # Same chunk returned by both vector search and expansion search
        shared_hit = _make_search_result("shared-1", "patterns-book/ch1.md", score=0.9)

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.side_effect = [
            [shared_hit],  # vector search returns the shared hit
            [shared_hit],  # expansion also returns it
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("cqrs pattern")

        # Deduplicated: only one result for shared-1
        chunk_ids = [r.get("source_path") for r in results]
        assert chunk_ids.count("patterns-book/ch1.md") == 1


# ---------------------------------------------------------------------------
# query_graph tool — _query_graph_impl
# ---------------------------------------------------------------------------


class TestQueryGraphKnownConcept:
    def test_known_concept_returns_node_and_neighbors(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="patterns-book"),
                _make_node("read-write-separation", source_slug="patterns-book"),
            ],
            [
                _make_edge("cqrs", "read-write-separation"),
                _make_edge("vertical-slice", "cqrs"),
            ],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs", depth=2)

        node_ids = {n["id"] for n in result["matched_nodes"]}
        assert "cqrs" in node_ids
        assert "read-write-separation" in node_ids
        assert "vertical-slice" in node_ids

    def test_relationships_included_for_known_concept(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [_make_node("cqrs"), _make_node("read-write-separation")],
            [_make_edge("cqrs", "read-write-separation", "enables")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs", depth=2)

        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["from"] == "cqrs"
        assert result["relationships"][0]["rel"] == "enables"
        assert result["relationships"][0]["to"] == "read-write-separation"

    def test_linked_sources_collected_from_neighborhood(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", source_slug="patterns-book"),
                _make_node("vertical-slice", source_slug="another-book"),
            ],
            [_make_edge("vertical-slice", "cqrs")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs", depth=2)

        assert "patterns-book" in result["linked_sources"]
        assert "another-book" in result["linked_sources"]

    def test_summary_is_non_empty_string(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [_make_node("cqrs", domain="architecture", node_type="pattern")],
            [],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs")

        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10
        assert "cqrs" in result["summary"]

    def test_summary_includes_domain_when_present(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [_make_node("cqrs", domain="architecture")],
            [],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs")

        assert "architecture" in result["summary"]

    def test_summary_includes_relationship_snippets(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [_make_node("cqrs"), _make_node("event-sourcing")],
            [_make_edge("cqrs", "event-sourcing", "enables")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs")

        assert "enables" in result["summary"]


class TestQueryGraphUnknownConcept:
    def test_unknown_concept_returns_empty_collections(self, temp_dir: Path) -> None:
        graph = _make_graph_store(temp_dir, [_make_node("cqrs")], [])

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("zzznomatch")

        assert result["matched_nodes"] == []
        assert result["relationships"] == []
        assert result["linked_sources"] == []

    def test_unknown_concept_returns_informative_summary(self, temp_dir: Path) -> None:
        graph = _make_graph_store(temp_dir, [_make_node("cqrs")], [])

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("zzznomatch")

        assert "zzznomatch" in result["summary"]

    def test_no_graph_returns_not_available_summary(self) -> None:
        with patch("grounded_code_mcp.server._get_graph_store", return_value=None):
            result = _query_graph_impl("anything")

        assert result["matched_nodes"] == []
        assert "No concept graph" in result["summary"]

    def test_empty_graph_returns_not_available_summary(self, temp_dir: Path) -> None:
        empty_store = GraphStore(path=temp_dir / "empty.json")
        empty_store.load()

        with patch("grounded_code_mcp.server._get_graph_store", return_value=empty_store):
            result = _query_graph_impl("anything")

        assert result["matched_nodes"] == []
        assert "No concept graph" in result["summary"]


class TestQueryGraphDomainFilter:
    def test_domain_filter_excludes_nodes_outside_domain(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", domain="architecture"),
                _make_node("tdd", domain="testing"),
            ],
            [_make_edge("cqrs", "tdd", "reinforces")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs", depth=2, domain="architecture")

        node_ids = {n["id"] for n in result["matched_nodes"]}
        assert "cqrs" in node_ids
        assert "tdd" not in node_ids

    def test_domain_filter_none_returns_full_neighborhood(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("cqrs", domain="architecture"),
                _make_node("tdd", domain="testing"),
            ],
            [_make_edge("cqrs", "tdd", "reinforces")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("cqrs", depth=2, domain=None)

        node_ids = {n["id"] for n in result["matched_nodes"]}
        assert "cqrs" in node_ids
        assert "tdd" in node_ids


class TestQueryGraphDepthLimit:
    def test_depth_1_does_not_return_two_hop_nodes(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("a"),
                _make_node("b"),
                _make_node("c"),
            ],
            [_make_edge("a", "b"), _make_edge("b", "c")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            result = _query_graph_impl("a", depth=1)

        node_ids = {n["id"] for n in result["matched_nodes"]}
        assert "b" in node_ids
        assert "c" not in node_ids

    def test_depth_clamped_to_max_of_3(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("a"),
                _make_node("b"),
                _make_node("c"),
                _make_node("d"),
                _make_node("e"),
            ],
            [
                _make_edge("a", "b"),
                _make_edge("b", "c"),
                _make_edge("c", "d"),
                _make_edge("d", "e"),
            ],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            # depth=99 should be clamped to 3
            result = _query_graph_impl("a", depth=99)

        node_ids = {n["id"] for n in result["matched_nodes"]}
        # Within 3 hops from "a": b, c, d — not e (4 hops)
        assert "d" in node_ids
        assert "e" not in node_ids

    def test_depth_clamped_to_minimum_of_1(self, temp_dir: Path) -> None:
        graph = _make_graph_store(
            temp_dir,
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b")],
        )

        with patch("grounded_code_mcp.server._get_graph_store", return_value=graph):
            # depth=0 is clamped to 1
            result = _query_graph_impl("a", depth=0)

        node_ids = {n["id"] for n in result["matched_nodes"]}
        assert "a" in node_ids
        assert "b" in node_ids


# ---------------------------------------------------------------------------
# Integration: real NetworkX graph + mock Qdrant
# ---------------------------------------------------------------------------


class TestIntegrationHybridSearch:
    """Integration tests using a real GraphStore (NetworkX) with mock Qdrant."""

    def test_single_source_rebuild_only_updates_that_source(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        """Ingest --force for a single source rebuilds graph for that source only."""
        from unittest.mock import patch as _patch

        from grounded_code_mcp.ingest import IngestionPipeline

        source_dir = settings.knowledge_base.sources_dir / "my-source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")
        (source_dir / "RELATIONSHIPS.md").write_text(
            '"CQRS" -> enables -> "Read-Write-Separation"\n'
        )

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            _patch("graph.graph_store.GraphStore") as mock_store_cls,
            _patch("graph.graph_builder.build") as mock_build,
        ):
            mock_store_cls.return_value = MagicMock()
            pipeline.ingest(source_dir, force=True)

        # build() should have been called with the specific RELATIONSHIPS.md file
        mock_build.assert_called_once()
        called_path = mock_build.call_args[0][0]
        assert called_path == source_dir / "RELATIONSHIPS.md"

    def test_global_force_rebuilds_entire_graph(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """Ingest --force without path calls build() with sources_dir."""
        from unittest.mock import patch as _patch

        from grounded_code_mcp.ingest import IngestionPipeline

        (settings.knowledge_base.sources_dir / "doc.md").write_text("# Doc\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            _patch("graph.graph_store.GraphStore"),
            _patch("graph.graph_builder.build") as mock_build,
        ):
            pipeline.ingest(force=True)

        mock_build.assert_called_once()
        called_path = mock_build.call_args[0][0]
        assert called_path == settings.knowledge_base.sources_dir

    def test_missing_relationships_md_logs_warning_does_not_fail(
        self,
        settings: Settings,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing RELATIONSHIPS.md logs a warning and does not raise."""
        import logging
        from unittest.mock import patch as _patch

        from grounded_code_mcp.embeddings import EmbeddingResult
        from grounded_code_mcp.ingest import IngestionPipeline

        source_dir = settings.knowledge_base.sources_dir / "orphan"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Orphan\n\nContent.")

        embedder = MagicMock()
        embedder.ensure_ready.return_value = None
        embedder.embed_many.side_effect = lambda texts, **_kw: [
            EmbeddingResult(text=t, embedding=[0.1] * 128, model="test") for t in texts
        ]

        pipeline = IngestionPipeline(settings, embedder=embedder)

        with (
            _patch("graph.graph_store.GraphStore"),
            _patch("graph.graph_builder.build") as mock_build,
            caplog.at_level(logging.WARNING, logger="grounded_code_mcp.ingest"),
        ):
            stats = pipeline.ingest(source_dir, force=True)

        assert stats.success is True
        mock_build.assert_not_called()
        assert any("RELATIONSHIPS.md" in r.message for r in caplog.records)

    def test_end_to_end_expansion_adds_graph_sourced_results(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        """Full hybrid search: real graph returns neighbor source, mock Qdrant returns chunk."""
        graph = _make_graph_store(
            temp_dir,
            [
                _make_node("ownership", source_slug="rust-book"),
                _make_node("memory-safety", source_slug="rust-book"),
            ],
            [_make_edge("ownership", "memory-safety", "enables")],
        )

        vector_hit = _make_search_result("v-1", "python/doc.md", score=0.95)
        expansion_hit = _make_search_result("e-1", "rust-book/ch4.md", score=0.75)

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.side_effect = [
            [vector_hit],
            [vector_hit, expansion_hit],
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
            patch("grounded_code_mcp.server._get_graph_store", return_value=graph),
        ):
            results = _search_knowledge_impl("ownership model")

        assert any("[graph-expanded" in r["retrieval_type"] for r in results)
        expanded = [r for r in results if "[graph-expanded" in r["retrieval_type"]]
        assert expanded[0]["source_path"] == "rust-book/ch4.md"
