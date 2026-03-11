"""Manifest repair utilities for recovering from crash-induced gaps.

When the ingestion process crashes after writing data to Qdrant but before
saving the manifest, files appear perpetually untracked.  The functions in
this module rebuild manifest entries from disk metadata and Qdrant state so
that those files are correctly recorded without re-ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from grounded_code_mcp.manifest import SourceEntry, compute_sha256
from grounded_code_mcp.parser import SUPPORTED_EXTENSIONS, get_file_type

logger = logging.getLogger(__name__)


def build_repair_entries(
    *,
    sources_dir: Path,
    sub_dir: str,
    collection_name: str,
    qdrant_chunks: dict[str, list[str]],
) -> list[SourceEntry]:
    """Build manifest entries for files on disk using Qdrant chunk data.

    This is a pure function: it reads the filesystem for file hashes and
    metadata, but does not contact Qdrant.  The caller is responsible for
    supplying ``qdrant_chunks`` (obtained via Qdrant scroll/filter).

    Args:
        sources_dir: Root sources directory (e.g. ``project/sources``).
        sub_dir: Subdirectory under sources_dir (e.g. ``"dotnet"``).
        collection_name: Full collection name including prefix
            (e.g. ``"grounded_dotnet"``).
        qdrant_chunks: Mapping of ``relative_path -> [chunk_id, ...]``
            extracted from Qdrant.  Paths are relative to ``sources_dir``
            (e.g. ``"dotnet/some-book.pdf"``).

    Returns:
        List of :class:`SourceEntry` objects ready to be added to a manifest.
    """
    target_dir = sources_dir / sub_dir
    if not target_dir.is_dir():
        logger.warning("Directory does not exist: %s", target_dir)
        return []

    entries: list[SourceEntry] = []

    for file_path in sorted(target_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = str(file_path.relative_to(sources_dir))
        chunk_ids = qdrant_chunks.get(relative_path, [])

        entry = SourceEntry(
            path=relative_path,
            sha256=compute_sha256(file_path),
            collection=collection_name,
            file_type=get_file_type(file_path),
            chunk_count=len(chunk_ids),
            chunk_ids=chunk_ids,
        )
        entries.append(entry)
        logger.debug(
            "Built repair entry: %s (%d chunks)",
            relative_path,
            len(chunk_ids),
        )

    return entries
