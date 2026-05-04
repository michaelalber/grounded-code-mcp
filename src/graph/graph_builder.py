"""Parse RELATIONSHIPS.md files and populate a GraphStore."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph.graph_store import GraphStore, slugify

logger = logging.getLogger(__name__)

# Quoted format: "Concept A" → rel → "Concept B" [opt1] [opt2] [opt3] [opt4]
# Supports both Unicode → and ASCII -> arrows.
_ARROW = r"(?:→|->)"
_TRIPLE_RE = re.compile(
    rf'"([^"]+)"\s*{_ARROW}\s*(\w+)\s*{_ARROW}\s*"([^"]+)"'
    r"(?:\s*\[([^\]]*)\])?"  # source_slug
    r"(?:\s*\[([^\]]*)\])?"  # domain
    r"(?:\s*\[([^\]]*)\])?"  # type
    r"(?:\s*\[([^\]]*)\])?"  # description
)

# Parenthetical format: (Concept A) --[PREDICATE]--> (Concept B) [opt1] ...
# Relations are uppercase in distilled sources; they are normalised to lowercase.
# Triples may appear inside fenced code blocks — fence markers are skipped silently.
_PAREN_TRIPLE_RE = re.compile(
    r"\(([^)]+)\)\s*--\[(\w+)\]-->\s*\(([^)]+)\)"
    r"(?:\s*\[([^\]]*)\])?"  # source_slug
    r"(?:\s*\[([^\]]*)\])?"  # domain
    r"(?:\s*\[([^\]]*)\])?"  # type
    r"(?:\s*\[([^\]]*)\])?"  # description
)


@dataclass
class BuildStats:
    """Results from a build run."""

    files_processed: int = 0
    triples_parsed: int = 0
    triples_skipped: int = 0
    nodes_added: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def _parse_triples(
    content: str,
    default_source_slug: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Parse concept triples from RELATIONSHIPS.md content.

    Supports two formats:
    - Quoted:        "Concept A" → rel → "Concept B" [source] [domain] [type] [desc]
    - Parenthetical: (Concept A) --[PREDICATE]--> (Concept B)

    Relations are normalised to lowercase. Fenced code block markers (```) are
    skipped silently. Any non-empty relation is accepted — no allowlist enforced.

    Returns (nodes, edges, skip_count). Nodes list contains one entry per
    unique concept id; duplicate appearances in the content are collapsed.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    skipped = 0
    seen_node_ids: set[str] = set()

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Fenced code block markers are structural — skip silently, not as malformed triples.
        if line.startswith("```") or line.startswith("~~~"):
            continue

        m = _TRIPLE_RE.search(line)
        if not m:
            m = _PAREN_TRIPLE_RE.search(line)
        if not m:
            logger.warning("Skipping malformed triple: %r", line)
            skipped += 1
            continue

        concept_a_raw, rel, concept_b_raw, raw_slug, raw_domain, raw_type, raw_desc = m.groups()

        rel = rel.strip().lower()

        node_a_id = slugify(concept_a_raw)
        node_b_id = slugify(concept_b_raw)
        source_slug = slugify(raw_slug) if raw_slug and raw_slug.strip() else default_source_slug
        domain = raw_domain.strip() if raw_domain and raw_domain.strip() else ""
        node_type = raw_type.strip() if raw_type and raw_type.strip() else ""
        description = raw_desc.strip() if raw_desc and raw_desc.strip() else ""

        for node_id in (node_a_id, node_b_id):
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                nodes.append(
                    {
                        "id": node_id,
                        "source_slug": source_slug,
                        "domain": domain,
                        "type": node_type,
                        "description": description,
                    }
                )

        edges.append({"source": node_a_id, "target": node_b_id, "rel": rel})

    return nodes, edges, skipped


def build(
    input_path: Path,
    store: GraphStore,
    *,
    dry_run: bool = False,
) -> BuildStats:
    """Parse all RELATIONSHIPS.md files under input_path and populate store.

    Args:
        input_path: Directory to search (or a single RELATIONSHIPS.md file).
        store: GraphStore to populate. Caller is responsible for load() first.
        dry_run: If True, parse and validate without writing to disk.

    Returns:
        BuildStats with file and triple counts.
    """
    stats = BuildStats(dry_run=dry_run)

    if not input_path.exists():
        return stats

    if input_path.is_file() and input_path.name == "RELATIONSHIPS.md":
        rel_files = [input_path]
    else:
        rel_files = sorted(input_path.rglob("RELATIONSHIPS.md"))

    for rel_file in rel_files:
        stats.files_processed += 1
        source_slug = slugify(rel_file.parent.name)
        content = rel_file.read_text(encoding="utf-8")
        nodes, edges, skipped = _parse_triples(content, default_source_slug=source_slug)
        stats.triples_parsed += len(edges)
        stats.triples_skipped += skipped

        if not dry_run:
            store.remove_source(source_slug)
            store.merge_nodes(nodes, edges)
            stats.nodes_added += len(nodes)

    if not dry_run and stats.files_processed > 0:
        store.save()

    return stats


def main() -> None:
    """CLI entry point: python -m graph.graph_builder --input <path> [--dry-run]."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build concept graph from RELATIONSHIPS.md files.")
    parser.add_argument(
        "--input", required=True, type=Path, help="Path containing RELATIONSHIPS.md files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate triples without writing the graph.",
    )
    args = parser.parse_args()

    store = GraphStore()
    store.load()

    stats = build(args.input, store, dry_run=args.dry_run)

    prefix = "[dry-run] " if args.dry_run else ""
    print(  # noqa: T201
        f"{prefix}{stats.files_processed} files, "
        f"{stats.triples_parsed} triples, "
        f"{stats.triples_skipped} skipped"
        + ("" if args.dry_run else f", {stats.nodes_added} nodes added")
    )


if __name__ == "__main__":
    main()
