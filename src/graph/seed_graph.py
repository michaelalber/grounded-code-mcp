"""Bootstrap RELATIONSHIPS.md files for source directories that lack them.

Generates starter relationship triples for well-known engineering domains.
Generated files are marked with a SEED header so reviewers know they need
human verification before being treated as authoritative.

CLI:
    python -m graph.seed_graph [--dry-run] [--source <slug>]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from graph.graph_store import slugify

logger = logging.getLogger(__name__)

SEED_HEADER = "<!-- SEED: auto-generated, review before committing -->"

# ---------------------------------------------------------------------------
# Domain-specific seed content for well-known collection slugs
# ---------------------------------------------------------------------------

_SEED_CONTENT: dict[str, str] = {
    "internal": """\
## Engineering Practices  <!-- domain: testing -->

(test-driven-development) --[enables]--> (refactoring)
(red-green-refactor) --[is-an-example-of]--> (test-driven-development)
(continuous-integration) --[enables]--> (fast-feedback)
(trunk-based-development) --[enables]--> (continuous-integration)
(pair-programming) --[enables]--> (knowledge-sharing)

## Architecture  <!-- domain: architecture -->

(clean-architecture) --[depends-on]--> (dependency-inversion)
(dependency-inversion) --[enables]--> (testability)
(yagni) --[conflicts-with]--> (speculative-abstraction)
(simple-design) --[improves]--> (test-driven-development)
(boy-scout-rule) --[improves]--> (simple-design)
""",
    "patterns": """\
## Structural Patterns  <!-- domain: architecture -->

(cqrs) --[enables]--> (read-write-separation)
(vertical-slice) --[enables]--> (cqrs)
(mediator) --[enables]--> (loose-coupling)
(decorator) --[is-an-example-of]--> (open-closed-principle)
(factory-method) --[enables]--> (dependency-inversion)
(observer) --[enables]--> (event-driven-architecture)
(strategy) --[enables]--> (behavioral-flexibility)
(domain-events) --[enables]--> (eventual-consistency)
""",
    "architecture": """\
## Cloud-Native  <!-- domain: architecture -->

(twelve-factor-app) --[enables]--> (cloud-native-portability)
(circuit-breaker) --[enables]--> (fault-tolerance)
(bulkhead) --[enables]--> (blast-radius-reduction)
(strangler-fig) --[enables]--> (incremental-migration)

## Reliability  <!-- domain: architecture -->

(service-level-objective) --[depends-on]--> (service-level-indicator)
(chaos-engineering) --[enables]--> (resilience-validation)
(distributed-tracing) --[enables]--> (observability)
(blue-green-deployment) --[enables]--> (zero-downtime)

## Data  <!-- domain: architecture -->

(event-sourcing) --[enables]--> (audit-log)
(saga-pattern) --[enables]--> (distributed-transaction)
""",
    "dotnet": """\
## Persistence  <!-- domain: dotnet -->

(entity-framework-core) --[enables]--> (orm-persistence)
(dbcontext) --[depends-on]--> (entity-framework-core)
(migrations) --[depends-on]--> (dbcontext)

## Design  <!-- domain: dotnet -->

(minimal-api) --[enables]--> (low-overhead-http-endpoints)
(dependency-injection) --[enables]--> (testability)
(ioptions) --[depends-on]--> (dependency-injection)
(vertical-slice-architecture) --[enables]--> (feature-cohesion)
(mediatr) --[is-an-example-of]--> (mediator)

## Safety  <!-- domain: dotnet -->

(cancellation-token) --[enables]--> (cooperative-cancellation)
(nullable-reference-types) --[enables]--> (null-safety)
""",
    "python": """\
## Web Frameworks  <!-- domain: python -->

(fastapi) --[enables]--> (type-safe-http-api)
(pydantic) --[enables]--> (runtime-validation)
(pydantic) --[depends-on]--> (type-annotations)
(fastmcp) --[is-an-example-of]--> (model-context-protocol)

## Testing and Quality  <!-- domain: python -->

(pytest) --[enables]--> (test-driven-development)
(ruff) --[enables]--> (fast-linting)
(mypy) --[enables]--> (static-type-checking)

## Design  <!-- domain: python -->

(asyncio) --[enables]--> (concurrent-io)
(dependency-injection) --[enables]--> (testability)
(pyproject-toml) --[enables]--> (reproducible-builds)
""",
    "databases": """\
## Indexing  <!-- domain: databases -->

(index) --[enables]--> (query-performance)
(b-tree-index) --[is-an-example-of]--> (index)
(covering-index) --[enables]--> (index-only-scan)

## Data Integrity  <!-- domain: databases -->

(normalisation) --[enables]--> (data-integrity)
(transaction) --[enables]--> (acid-guarantees)
(write-ahead-log) --[enables]--> (crash-recovery)

## Scalability  <!-- domain: databases -->

(replication) --[enables]--> (high-availability)
(partitioning) --[enables]--> (horizontal-scalability)
(connection-pooling) --[enables]--> (resource-efficiency)

## Security  <!-- domain: databases -->

(prepared-statement) --[prevents]--> (sql-injection)
""",
    "rust": """\
## Memory Safety  <!-- domain: rust -->

(ownership) --[enables]--> (memory-safety)
(borrow-checker) --[depends-on]--> (ownership)
(lifetimes) --[enables]--> (reference-validity)

## Type System  <!-- domain: rust -->

(result) --[enables]--> (explicit-error-handling)
(option) --[enables]--> (null-safety)
(traits) --[enables]--> (polymorphism)

## Ecosystem  <!-- domain: rust -->

(async-await) --[enables]--> (concurrent-io)
(tokio) --[is-an-example-of]--> (async-await)
(cargo) --[enables]--> (reproducible-builds)
(axum) --[enables]--> (type-safe-http-api)
""",
    "edge-ai": """\
## RAG Pipeline  <!-- domain: edge-ai -->

(rag) --[enables]--> (grounded-responses)
(vector-store) --[enables]--> (semantic-search)
(embedding-model) --[depends-on]--> (vector-store)
(chunking) --[enables]--> (retrieval-precision)
(semantic-similarity) --[depends-on]--> (embedding-model)
(reranking) --[enables]--> (search-result-quality)

## Quality  <!-- domain: edge-ai -->

(hallucination) --[conflicts-with]--> (grounded-responses)
(eval-suite) --[enables]--> (output-quality-gate)
(prompt-template) --[enables]--> (reproducible-prompting)

## Architecture  <!-- domain: edge-ai -->

(mcp) --[enables]--> (tool-augmented-llm)
""",
}


def _generic_template(source_slug: str) -> str:
    """Return a minimal starter template for unknown source slugs."""
    return (
        "## Core Concepts  <!-- domain: architecture -->\n\n"
        "(concept-a) --[enables]--> (concept-b)\n"
        "(concept-b) --[depends-on]--> (concept-c)\n"
        "# TODO: replace placeholder concepts with real relationships from this source\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_sources_missing_relationships(sources_dir: Path) -> list[Path]:
    """Return all source directories under sources_dir that lack RELATIONSHIPS.md.

    A directory qualifies as a source if it contains at least one file that is
    not RELATIONSHIPS.md itself.  The RELATIONSHIPS.md file is excluded from
    that count so that orphaned-relationships dirs are not treated as sources.
    """
    if not sources_dir.exists():
        return []

    source_dirs: set[Path] = {
        f.parent for f in sources_dir.rglob("*") if f.is_file() and f.name != "RELATIONSHIPS.md"
    }

    return sorted(d for d in source_dirs if not (d / "RELATIONSHIPS.md").exists())


def generate_seed_content(source_slug: str) -> str:
    """Return seed RELATIONSHIPS.md content for source_slug.

    Well-known slugs get domain-specific relationships; unknown slugs get a
    generic placeholder template.
    """
    body = _SEED_CONTENT.get(source_slug) or _generic_template(source_slug)
    return f"{SEED_HEADER}\n\n{body}"


@dataclass
class SeedStats:
    """Results from a seed run."""

    sources_found: int = 0
    files_written: int = 0
    files_skipped: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def seed(
    sources_dir: Path,
    *,
    source: str | None = None,
    dry_run: bool = False,
) -> SeedStats:
    """Generate seed RELATIONSHIPS.md files for sources that lack them.

    Args:
        sources_dir: Root directory containing source sub-directories.
        source: If provided, target only the named source slug.
        dry_run: When True, report what would be written without writing.

    Returns:
        SeedStats with counts of sources found and files written.

    Raises:
        ValueError: If source is specified but not found in sources_dir.
    """
    stats = SeedStats(dry_run=dry_run)

    if source is not None:
        target_dir = _resolve_source_dir(sources_dir, source)
        if target_dir is None:
            raise ValueError(
                f"Source '{source}' not found under {sources_dir}. "
                "Run without --source to see available sources."
            )
        candidates = [target_dir] if not (target_dir / "RELATIONSHIPS.md").exists() else []
    else:
        candidates = find_sources_missing_relationships(sources_dir)

    stats.sources_found = len(candidates)

    for source_dir in candidates:
        source_slug = slugify(source_dir.name)
        rel_file = source_dir / "RELATIONSHIPS.md"

        if dry_run:
            logger.info("[dry-run] Would write %s", rel_file)
            continue

        try:
            content = generate_seed_content(source_slug)
            rel_file.write_text(content, encoding="utf-8")
            stats.files_written += 1
            logger.info("Seeded %s", rel_file)
        except OSError as exc:
            logger.error("Failed to write %s: %s", rel_file, exc)
            stats.errors.append(f"{rel_file}: {exc}")

    return stats


def _resolve_source_dir(sources_dir: Path, source_slug: str) -> Path | None:
    """Find the directory for a given source slug under sources_dir."""
    for candidate in sources_dir.rglob("*"):
        if candidate.is_dir() and slugify(candidate.name) == source_slug:
            source_files = [
                f for f in candidate.iterdir() if f.is_file() and f.name != "RELATIONSHIPS.md"
            ]
            if source_files:
                return candidate
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: python -m graph.seed_graph [--dry-run] [--source <slug>]."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate starter RELATIONSHIPS.md for sources that lack them."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sources missing RELATIONSHIPS.md without writing any files.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Seed only this specific source slug.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path("sources"),
        help="Root sources directory (default: ./sources).",
    )
    args = parser.parse_args()

    try:
        stats = seed(args.sources_dir, source=args.source, dry_run=args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))

    prefix = "[dry-run] " if args.dry_run else ""
    if args.dry_run:
        print(f"{prefix}{stats.sources_found} sources need seeding")  # noqa: T201
    else:
        print(  # noqa: T201
            f"{stats.files_written} files written, "
            f"{stats.sources_found - stats.files_written} skipped"
        )


if __name__ == "__main__":
    main()
