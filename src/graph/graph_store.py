"""Concept graph backed by a NetworkX DiGraph, persisted as JSON."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph as nx_json

logger = logging.getLogger(__name__)

_DEFAULT_GRAPH_PATH = Path("graph") / "concept_graph.json"

VALID_DOMAINS: frozenset[str] = frozenset(
    {
        "architecture",
        "testing",
        "data-access",
        "agent-behavior",
        "quality",
        "patterns",
        "constraints",
    }
)

VALID_TYPES: frozenset[str] = frozenset(
    {
        "pattern",
        "principle",
        "practice",
        "anti-pattern",
        "constraint",
    }
)

VALID_RELATIONS: frozenset[str] = frozenset(
    {
        "depends_on",
        "enables",
        "conflicts_with",
        "is_example_of",
        "reinforces",
    }
)


def slugify(text: str) -> str:
    """Convert concept text to a slug: lowercase, hyphens, no special characters."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


class GraphStore:
    """Concept graph backed by a NetworkX DiGraph, persisted as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.environ.get("GRAPH_JSON_PATH")
        self._path: Path = (
            Path(env_path) if env_path else (path if path is not None else _DEFAULT_GRAPH_PATH)
        )
        self._graph: Any = nx.DiGraph()

    # ------------------------------------------------------------------
    # Convenience properties used in tests and CLI
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def node_count(self) -> int:
        return int(self._graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self._graph.number_of_edges())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load graph from JSON. Creates an empty graph when the file is absent."""
        if not self._path.exists():
            self._graph = nx.DiGraph()
            return
        with open(self._path) as f:
            data: dict[str, Any] = json.load(f)
        self._graph = nx_json.node_link_graph(data, directed=True, multigraph=False, edges="edges")

    def save(self) -> None:
        """Persist graph to JSON, creating parent directories as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx_json.node_link_data(self._graph, edges="edges")
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_to_dict(self, node_id: str) -> dict[str, Any]:
        attrs: dict[str, Any] = dict(self._graph.nodes[node_id])
        return {"id": node_id, **attrs}

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_neighbors(self, concept_id: str, depth: int = 2) -> list[dict[str, Any]]:
        """Return all nodes reachable (in or out) from concept_id within depth hops."""
        if concept_id not in self._graph:
            return []
        visited: set[str] = {concept_id}
        frontier: set[str] = {concept_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited
            visited.update(frontier)
        visited.discard(concept_id)
        return [self._node_to_dict(n) for n in visited if n in self._graph]

    def get_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Return all nodes with the given domain."""
        return [
            self._node_to_dict(n)
            for n, attrs in self._graph.nodes(data=True)
            if attrs.get("domain") == domain
        ]

    def get_by_source(self, source_slug: str) -> list[dict[str, Any]]:
        """Return all nodes with the given source_slug."""
        return [
            self._node_to_dict(n)
            for n, attrs in self._graph.nodes(data=True)
            if attrs.get("source_slug") == source_slug
        ]

    def find_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Return the shortest directed path between two nodes, or None."""
        try:
            return list(nx.shortest_path(self._graph, from_id, to_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def search_nodes(self, query_term: str) -> list[dict[str, Any]]:
        """Case-insensitive substring search on node id and description."""
        term = query_term.lower()
        return [
            self._node_to_dict(n)
            for n, attrs in self._graph.nodes(data=True)
            if term in n or term in (attrs.get("description") or "").lower()
        ]

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def remove_source(self, source_slug: str) -> None:
        """Delete all nodes for source_slug and their attached edges. Idempotent."""
        to_remove = [
            n
            for n, attrs in self._graph.nodes(data=True)
            if attrs.get("source_slug") == source_slug
        ]
        self._graph.remove_nodes_from(to_remove)

    def merge_nodes(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        """Additive upsert of nodes and edges.

        Node dicts must have an "id" key. Edge dicts must have
        "source", "target", and "rel" keys. Existing nodes are updated
        with new attributes; existing edges have their "rel" attribute updated.
        """
        for node in nodes:
            node_id: str = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            if self._graph.has_node(node_id):
                self._graph.nodes[node_id].update(attrs)
            else:
                self._graph.add_node(node_id, **attrs)
        for edge in edges:
            src: str = edge["source"]
            dst: str = edge["target"]
            rel: str = edge.get("rel", "")
            self._graph.add_edge(src, dst, rel=rel)
