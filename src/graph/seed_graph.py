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
"Test-Driven Development" → enables → "Refactoring" [internal] [testing] [practice]
"Red-Green-Refactor" → is_example_of → "Test-Driven Development" [internal] [testing] [practice]
"Clean Architecture" → depends_on → "Dependency Inversion" [internal] [architecture] [principle]
"Dependency Inversion" → enables → "Testability" [internal] [architecture] [principle]
"Continuous Integration" → enables → "Fast Feedback" [internal] [quality] [practice]
"Trunk-Based Development" → enables → "Continuous Integration" [internal] [quality] [practice]
"YAGNI" → conflicts_with → "Speculative Abstraction" [internal] [quality] [principle]
"Simple Design" → reinforces → "Test-Driven Development" [internal] [quality] [principle]
"Boy Scout Rule" → reinforces → "Simple Design" [internal] [quality] [principle]
"Pair Programming" → enables → "Knowledge Sharing" [internal] [quality] [practice]
""",
    "patterns": """\
"CQRS" → enables → "Read-Write Separation" [patterns] [architecture] [pattern]
"Vertical Slice" → enables → "CQRS" [patterns] [architecture] [pattern]
"Repository Pattern" → enables → "Data Access Abstraction" [patterns] [data-access] [pattern]
"Unit of Work" → depends_on → "Repository Pattern" [patterns] [data-access] [pattern]
"Mediator" → enables → "Loose Coupling" [patterns] [patterns] [pattern]
"Decorator" → is_example_of → "Open-Closed Principle" [patterns] [patterns] [pattern]
"Factory Method" → enables → "Dependency Inversion" [patterns] [patterns] [pattern]
"Observer" → enables → "Event-Driven Architecture" [patterns] [patterns] [pattern]
"Strategy" → enables → "Behavioral Flexibility" [patterns] [patterns] [pattern]
"Domain Events" → enables → "Eventual Consistency" [patterns] [architecture] [pattern]
""",
    "architecture": """\
"12-Factor App" → enables → "Cloud-Native Portability" [architecture] [architecture] [principle]
"Service Level Objective" → depends_on → "Service Level Indicator" [architecture] [architecture] [principle]
"Circuit Breaker" → enables → "Fault Tolerance" [architecture] [architecture] [pattern]
"Bulkhead" → enables → "Blast Radius Reduction" [architecture] [architecture] [pattern]
"Event Sourcing" → enables → "Audit Log" [architecture] [architecture] [pattern]
"Saga Pattern" → enables → "Distributed Transaction" [architecture] [architecture] [pattern]
"Strangler Fig" → enables → "Incremental Migration" [architecture] [architecture] [pattern]
"Blue-Green Deployment" → enables → "Zero Downtime" [architecture] [architecture] [practice]
"Chaos Engineering" → enables → "Resilience Validation" [architecture] [architecture] [practice]
"Distributed Tracing" → enables → "Observability" [architecture] [architecture] [practice]
""",
    "dotnet": """\
"Entity Framework Core" → enables → "ORM Persistence" [dotnet] [data-access] [practice]
"DbContext" → depends_on → "Entity Framework Core" [dotnet] [data-access] [practice]
"Migrations" → depends_on → "DbContext" [dotnet] [data-access] [practice]
"Minimal API" → enables → "Low-Overhead HTTP Endpoints" [dotnet] [architecture] [practice]
"Dependency Injection" → enables → "Testability" [dotnet] [architecture] [principle]
"IOptions" → depends_on → "Dependency Injection" [dotnet] [architecture] [practice]
"CancellationToken" → enables → "Cooperative Cancellation" [dotnet] [quality] [practice]
"Nullable Reference Types" → enables → "Null Safety" [dotnet] [quality] [practice]
"Vertical Slice Architecture" → enables → "Feature Cohesion" [dotnet] [architecture] [pattern]
"MediatR" → is_example_of → "Mediator" [dotnet] [architecture] [pattern]
""",
    "python": """\
"FastAPI" → enables → "Type-Safe HTTP API" [python] [architecture] [practice]
"Pydantic" → enables → "Runtime Validation" [python] [quality] [practice]
"Pydantic" → depends_on → "Type Annotations" [python] [quality] [practice]
"pytest" → enables → "Test-Driven Development" [python] [testing] [practice]
"asyncio" → enables → "Concurrent I/O" [python] [quality] [practice]
"FastMCP" → is_example_of → "Model Context Protocol" [python] [architecture] [practice]
"Dependency Injection" → enables → "Testability" [python] [architecture] [principle]
"pyproject.toml" → enables → "Reproducible Builds" [python] [quality] [practice]
"Ruff" → enables → "Fast Linting" [python] [quality] [practice]
"mypy" → enables → "Static Type Checking" [python] [quality] [practice]
""",
    "databases": """\
"Index" → enables → "Query Performance" [databases] [data-access] [practice]
"B-Tree Index" → is_example_of → "Index" [databases] [data-access] [pattern]
"Covering Index" → enables → "Index-Only Scan" [databases] [data-access] [pattern]
"Normalisation" → enables → "Data Integrity" [databases] [data-access] [principle]
"Transaction" → enables → "ACID Guarantees" [databases] [data-access] [principle]
"Write-Ahead Log" → enables → "Crash Recovery" [databases] [data-access] [pattern]
"Replication" → enables → "High Availability" [databases] [data-access] [pattern]
"Partitioning" → enables → "Horizontal Scalability" [databases] [data-access] [pattern]
"Connection Pooling" → enables → "Resource Efficiency" [databases] [data-access] [practice]
"Prepared Statement" → enables → "SQL Injection Prevention" [databases] [data-access] [practice]
""",
    "rust": """\
"Ownership" → enables → "Memory Safety" [rust] [quality] [principle]
"Borrow Checker" → depends_on → "Ownership" [rust] [quality] [principle]
"Lifetimes" → enables → "Reference Validity" [rust] [quality] [principle]
"Traits" → enables → "Polymorphism" [rust] [patterns] [pattern]
"async/await" → enables → "Concurrent I/O" [rust] [quality] [practice]
"Tokio" → is_example_of → "async/await" [rust] [quality] [practice]
"Result" → enables → "Explicit Error Handling" [rust] [quality] [principle]
"Option" → enables → "Null Safety" [rust] [quality] [principle]
"Cargo" → enables → "Reproducible Builds" [rust] [quality] [practice]
"Axum" → enables → "Type-Safe HTTP API" [rust] [architecture] [practice]
""",
    "edge-ai": """\
"RAG" → enables → "Grounded Responses" [edge-ai] [architecture] [pattern]
"Vector Store" → enables → "Semantic Search" [edge-ai] [architecture] [pattern]
"Embedding Model" → depends_on → "Vector Store" [edge-ai] [architecture] [practice]
"Chunking" → enables → "Retrieval Precision" [edge-ai] [architecture] [practice]
"Semantic Similarity" → depends_on → "Embedding Model" [edge-ai] [architecture] [principle]
"Reranking" → enables → "Search Result Quality" [edge-ai] [architecture] [practice]
"Hallucination" → conflicts_with → "Grounded Responses" [edge-ai] [quality] [anti-pattern]
"MCP" → enables → "Tool-Augmented LLM" [edge-ai] [architecture] [pattern]
"Prompt Template" → enables → "Reproducible Prompting" [edge-ai] [quality] [practice]
"Eval Suite" → enables → "Output Quality Gate" [edge-ai] [quality] [practice]
""",
}


def _generic_template(source_slug: str) -> str:
    """Return a minimal starter template for unknown source slugs."""
    return (
        f'"[Concept A]" → enables → "[Concept B]" [{source_slug}] [architecture] [principle]\n'
        f'"[Concept B]" → depends_on → "[Concept C]" [{source_slug}] [architecture] [principle]\n'
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
        f.parent
        for f in sources_dir.rglob("*")
        if f.is_file() and f.name != "RELATIONSHIPS.md"
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
