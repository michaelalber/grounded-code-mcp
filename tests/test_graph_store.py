"""Tests for GraphStore: persistence, queries, remove_source, merge_nodes."""

from pathlib import Path
from typing import Any

import pytest

from graph.graph_store import GraphStore, slugify

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercases(self) -> None:
        assert slugify("Vertical Slice") == "vertical-slice"

    def test_replaces_spaces_with_hyphens(self) -> None:
        assert slugify("red green refactor") == "red-green-refactor"

    def test_strips_special_characters(self) -> None:
        assert slugify("CQRS (Command/Query)") == "cqrs-commandquery"

    def test_collapses_multiple_hyphens(self) -> None:
        assert slugify("test--double") == "test-double"

    def test_strips_leading_trailing_hyphens(self) -> None:
        assert slugify("  -test- ") == "test"

    def test_underscores_become_hyphens(self) -> None:
        assert slugify("some_concept") == "some-concept"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    concept_id: str,
    source_slug: str = "test-source",
    domain: str = "architecture",
    node_type: str = "pattern",
    description: str = "A test concept.",
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
# load / save
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_load_creates_empty_graph_when_no_file_exists(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "graph.json")
        store.load()
        assert store.node_count == 0
        assert store.edge_count == 0

    def test_save_creates_parent_directories(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "nested" / "dir" / "graph.json")
        store.load()
        store.save()
        assert (temp_dir / "nested" / "dir" / "graph.json").exists()

    def test_roundtrip_preserves_nodes(self, temp_dir: Path) -> None:
        path = temp_dir / "graph.json"
        store = GraphStore(path=path)
        store.load()
        store.merge_nodes([_make_node("vertical-slice"), _make_node("cqrs")], [])
        store.save()

        store2 = GraphStore(path=path)
        store2.load()
        assert store2.node_count == 2
        ids = {n["id"] for n in store2.get_by_source("test-source")}
        assert ids == {"vertical-slice", "cqrs"}

    def test_roundtrip_preserves_edges(self, temp_dir: Path) -> None:
        path = temp_dir / "graph.json"
        store = GraphStore(path=path)
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b", "enables")],
        )
        store.save()

        store2 = GraphStore(path=path)
        store2.load()
        assert store2.edge_count == 1

    def test_roundtrip_preserves_node_attributes(self, temp_dir: Path) -> None:
        path = temp_dir / "graph.json"
        store = GraphStore(path=path)
        store.load()
        store.merge_nodes(
            [_make_node("tidy-first", source_slug="tidy-first-book", domain="quality")],
            [],
        )
        store.save()

        store2 = GraphStore(path=path)
        store2.load()
        nodes = store2.get_by_source("tidy-first-book")
        assert len(nodes) == 1
        assert nodes[0]["domain"] == "quality"

    def test_env_var_overrides_path(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_path = temp_dir / "env_graph.json"
        monkeypatch.setenv("GRAPH_JSON_PATH", str(env_path))
        store = GraphStore()
        assert store.path == env_path


# ---------------------------------------------------------------------------
# remove_source
# ---------------------------------------------------------------------------


class TestRemoveSource:
    def test_removes_all_nodes_for_source(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [
                _make_node("a", source_slug="book-a"),
                _make_node("b", source_slug="book-a"),
                _make_node("c", source_slug="book-b"),
            ],
            [],
        )
        store.remove_source("book-a")
        assert store.node_count == 1
        assert store.get_by_source("book-b")[0]["id"] == "c"

    def test_idempotent_on_same_source(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("a", source_slug="book-a")], [])
        store.remove_source("book-a")
        store.remove_source("book-a")  # second call must not raise
        assert store.node_count == 0

    def test_idempotent_on_nonexistent_source(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.remove_source("nonexistent")  # must not raise
        assert store.node_count == 0

    def test_removes_edges_connected_to_removed_nodes(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a", source_slug="src"), _make_node("b", source_slug="other")],
            [_make_edge("a", "b")],
        )
        assert store.edge_count == 1
        store.remove_source("src")
        assert store.edge_count == 0

    def test_does_not_remove_nodes_from_other_sources(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [
                _make_node("a", source_slug="src-a"),
                _make_node("b", source_slug="src-b"),
            ],
            [],
        )
        store.remove_source("src-a")
        remaining = store.get_by_source("src-b")
        assert len(remaining) == 1
        assert remaining[0]["id"] == "b"


# ---------------------------------------------------------------------------
# merge_nodes
# ---------------------------------------------------------------------------


class TestMergeNodes:
    def test_adds_new_nodes(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("a"), _make_node("b")], [])
        assert store.node_count == 2

    def test_no_duplicates_on_repeated_call(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        nodes = [_make_node("a"), _make_node("b")]
        store.merge_nodes(nodes, [])
        store.merge_nodes(nodes, [])
        assert store.node_count == 2

    def test_updates_existing_node_attributes(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("a", domain="testing")], [])
        store.merge_nodes([_make_node("a", domain="architecture")], [])
        nodes = store.get_by_source("test-source")
        assert nodes[0]["domain"] == "architecture"

    def test_adds_edges(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b", "enables")],
        )
        assert store.edge_count == 1

    def test_no_duplicate_edges_on_repeated_call(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        nodes = [_make_node("a"), _make_node("b")]
        edge = [_make_edge("a", "b")]
        store.merge_nodes(nodes, edge)
        store.merge_nodes(nodes, edge)
        assert store.edge_count == 1

    def test_additive_does_not_delete_existing(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("a")], [])
        store.merge_nodes([_make_node("b")], [])
        assert store.node_count == 2


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def test_returns_direct_neighbors_at_depth_1(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b"), _make_node("c")],
            [_make_edge("a", "b"), _make_edge("b", "c")],
        )
        neighbors = {n["id"] for n in store.get_neighbors("a", depth=1)}
        assert neighbors == {"b"}

    def test_returns_transitive_neighbors_at_depth_2(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b"), _make_node("c")],
            [_make_edge("a", "b"), _make_edge("b", "c")],
        )
        neighbors = {n["id"] for n in store.get_neighbors("a", depth=2)}
        assert "b" in neighbors
        assert "c" in neighbors

    def test_returns_empty_for_unknown_node(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        assert store.get_neighbors("nonexistent") == []

    def test_includes_predecessors(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b")],
        )
        neighbors = {n["id"] for n in store.get_neighbors("b", depth=1)}
        assert "a" in neighbors


# ---------------------------------------------------------------------------
# get_by_domain / get_by_source
# ---------------------------------------------------------------------------


class TestGetByDomain:
    def test_filters_matching_domain(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [
                _make_node("a", domain="architecture"),
                _make_node("b", domain="testing"),
            ],
            [],
        )
        result = store.get_by_domain("architecture")
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_returns_empty_for_unknown_domain(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        assert store.get_by_domain("unknown-domain") == []


class TestGetBySource:
    def test_filters_matching_source_slug(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [
                _make_node("a", source_slug="book-a"),
                _make_node("b", source_slug="book-b"),
            ],
            [],
        )
        result = store.get_by_source("book-a")
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_returns_empty_for_unknown_source(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        assert store.get_by_source("nonexistent") == []


# ---------------------------------------------------------------------------
# find_path
# ---------------------------------------------------------------------------


class TestFindPath:
    def test_finds_direct_path(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b")],
        )
        path = store.find_path("a", "b")
        assert path == ["a", "b"]

    def test_finds_indirect_path(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b"), _make_node("c")],
            [_make_edge("a", "b"), _make_edge("b", "c")],
        )
        path = store.find_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_returns_none_when_no_directed_path(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a"), _make_node("b")],
            [_make_edge("a", "b")],
        )
        assert store.find_path("b", "a") is None

    def test_returns_none_for_unknown_node(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        assert store.find_path("x", "y") is None


# ---------------------------------------------------------------------------
# search_nodes
# ---------------------------------------------------------------------------


class TestSearchNodes:
    def test_matches_by_id_substring(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("vertical-slice"), _make_node("cqrs")],
            [],
        )
        results = store.search_nodes("vertical")
        assert len(results) == 1
        assert results[0]["id"] == "vertical-slice"

    def test_matches_by_description_substring(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes(
            [_make_node("a", description="A pattern for slicing features vertically.")],
            [],
        )
        results = store.search_nodes("slicing")
        assert len(results) == 1

    def test_case_insensitive(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("vertical-slice")], [])
        assert len(store.search_nodes("VERTICAL")) == 1

    def test_returns_empty_when_no_match(self, temp_dir: Path) -> None:
        store = GraphStore(path=temp_dir / "g.json")
        store.load()
        store.merge_nodes([_make_node("cqrs")], [])
        assert store.search_nodes("zzznomatch") == []
