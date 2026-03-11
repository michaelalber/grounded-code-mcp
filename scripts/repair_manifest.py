#!/usr/bin/env python3
"""Repair manifest entries for collections whose data exists in Qdrant but
was lost from the manifest due to process crashes during ingestion.

Usage:
    python scripts/repair_manifest.py [--dry-run] [COLLECTION ...]

Examples:
    # Preview what would change (no writes):
    python scripts/repair_manifest.py --dry-run

    # Repair specific collections:
    python scripts/repair_manifest.py dotnet automation

    # Repair all collections that have untracked files:
    python scripts/repair_manifest.py
"""
# ruff: noqa: T201  — print is intentional in CLI scripts

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from grounded_code_mcp.config import Settings  # noqa: E402
from grounded_code_mcp.manifest import Manifest  # noqa: E402
from grounded_code_mcp.manifest_repair import build_repair_entries  # noqa: E402

logger = logging.getLogger(__name__)

# Upper bound per scroll page — Qdrant default is 10, we want fewer round-trips.
_SCROLL_LIMIT = 250


def _scroll_chunk_ids(
    qdrant_url: str,
    collection_name: str,
) -> dict[str, list[str]]:
    """Scroll all points in *collection_name* and return ``{source_path: [id, …]}``."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    result: dict[str, list[str]] = {}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=["source_path"],
            with_vectors=False,
        )
        for pt in points:
            source_path = (pt.payload or {}).get("source_path", "")
            if source_path:
                result.setdefault(source_path, []).append(str(pt.id))

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    return result


def _resolve_qdrant_url(settings: Settings) -> str:
    """Return the Qdrant HTTP URL from settings, falling back to localhost."""
    return settings.vectorstore.qdrant_url or "http://localhost:6333"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair manifest entries from Qdrant state.")
    parser.add_argument(
        "collections",
        nargs="*",
        help="Collection suffixes to repair (e.g. 'dotnet automation'). "
        "If omitted, repairs all collections with untracked files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be repaired without writing the manifest.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load settings (project + user config)
    settings = Settings.load(config_path=_PROJECT_DIR / "config.toml")
    sources_dir = _PROJECT_DIR / settings.knowledge_base.sources_dir
    manifest_path = _PROJECT_DIR / settings.knowledge_base.manifest_path
    qdrant_url = _resolve_qdrant_url(settings)

    # Load manifest
    manifest = Manifest.load_or_create(manifest_path)
    original_count = len(manifest.sources)

    # Determine which collections to repair
    prefix = settings.vectorstore.collection_prefix
    if args.collections:
        targets = {suffix: f"{prefix}{suffix}" for suffix in args.collections}
    else:
        # Auto-detect: repair any collection whose source dir has files but
        # the manifest has no entries for that prefix.
        targets = {}
        for src_dir_str, suffix in settings.collections.items():
            # src_dir_str is e.g. "sources/dotnet"
            sub_dir = Path(src_dir_str).relative_to(settings.knowledge_base.sources_dir)
            dir_prefix = str(sub_dir).rstrip("/") + "/"
            tracked = sum(1 for path in manifest.sources if path.startswith(dir_prefix))
            disk_dir = _PROJECT_DIR / src_dir_str
            if disk_dir.is_dir() and tracked == 0:
                targets[str(sub_dir)] = f"{prefix}{suffix}"

    if not targets:
        print("Nothing to repair — all collections have manifest entries.")
        return

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Manifest:   {manifest_path} ({original_count} entries)")
    print(f"Repairing:  {', '.join(f'{k} → {v}' for k, v in targets.items())}")
    print()

    total_entries = 0
    total_chunks = 0

    for sub_dir, collection_name in targets.items():
        print(f"── {collection_name} ({sub_dir}/) ──")

        # Scroll Qdrant for all chunk IDs grouped by source_path
        print(f"  Scrolling Qdrant collection {collection_name}...")
        qdrant_chunks = _scroll_chunk_ids(qdrant_url, collection_name)
        print(
            f"  Found {sum(len(v) for v in qdrant_chunks.values())} chunks "
            f"across {len(qdrant_chunks)} source paths"
        )

        # Build repair entries
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir=sub_dir,
            collection_name=collection_name,
            qdrant_chunks=qdrant_chunks,
        )

        if not entries:
            print("  No files found on disk — skipping.")
            continue

        for entry in entries:
            status = "NEW" if manifest.get_entry(entry.path) is None else "UPDATE"
            print(f"  {status}: {entry.path} ({entry.chunk_count} chunks)")
            if not args.dry_run:
                manifest.add_entry(entry)

        total_entries += len(entries)
        total_chunks += sum(e.chunk_count for e in entries)
        print()

    if args.dry_run:
        print(
            f"DRY RUN: would add {total_entries} entries ({total_chunks} chunks) to the manifest."
        )
    else:
        manifest.save(manifest_path)
        print(
            f"Saved manifest: {len(manifest.sources)} entries "
            f"(was {original_count}, added {total_entries})"
        )
        print(f"Total repaired chunks: {total_chunks}")


if __name__ == "__main__":
    main()
