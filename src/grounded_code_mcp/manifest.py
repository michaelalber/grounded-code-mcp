"""Manifest for tracking ingested documents and their chunks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    chunk_index: int = Field(description="Index of chunk within the source document")
    start_char: int = Field(default=0, description="Start character offset in source")
    end_char: int = Field(default=0, description="End character offset in source")
    heading_context: list[str] = Field(
        default_factory=list,
        description="Hierarchy of headings leading to this chunk",
    )
    is_code: bool = Field(default=False, description="Whether this chunk is a code block")
    code_language: str | None = Field(default=None, description="Programming language if code")
    is_table: bool = Field(default=False, description="Whether this chunk is a table")


class SourceEntry(BaseModel):
    """Entry tracking an ingested source document."""

    path: str = Field(description="Relative path to the source file")
    sha256: str = Field(description="SHA-256 hash of the file contents")
    collection: str = Field(description="Vector store collection name")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the file was ingested",
    )
    file_type: str = Field(default="", description="Detected file type (pdf, docx, md, etc.)")
    title: str | None = Field(default=None, description="Extracted document title")
    page_count: int | None = Field(default=None, description="Number of pages if applicable")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of chunk IDs in the vector store",
    )

    def has_changed(self, new_hash: str) -> bool:
        """Check if the source file has changed.

        Args:
            new_hash: New SHA-256 hash to compare.

        Returns:
            True if the hash differs from stored hash.
        """
        return self.sha256 != new_hash


class Manifest(BaseModel):
    """Manifest tracking all ingested documents."""

    version: str = Field(default="1.0", description="Manifest schema version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the manifest was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the manifest was last updated",
    )
    sources: dict[str, SourceEntry] = Field(
        default_factory=dict,
        description="Map of relative paths to source entries",
    )

    def get_entry(self, path: Path | str) -> SourceEntry | None:
        """Get the entry for a source file.

        Args:
            path: Path to the source file.

        Returns:
            SourceEntry if found, None otherwise.
        """
        key = str(path)
        return self.sources.get(key)

    def add_entry(self, entry: SourceEntry) -> None:
        """Add or update a source entry.

        Args:
            entry: SourceEntry to add or update.
        """
        self.sources[entry.path] = entry
        self.updated_at = datetime.now(timezone.utc)

    def remove_entry(self, path: Path | str) -> SourceEntry | None:
        """Remove a source entry.

        Args:
            path: Path to remove.

        Returns:
            Removed SourceEntry if found, None otherwise.
        """
        key = str(path)
        entry = self.sources.pop(key, None)
        if entry:
            self.updated_at = datetime.now(timezone.utc)
        return entry

    def needs_reingestion(self, path: Path) -> bool:
        """Check if a file needs to be re-ingested.

        Args:
            path: Path to the source file.

        Returns:
            True if file is new or has changed.
        """
        entry = self.get_entry(path)
        if entry is None:
            return True

        current_hash = compute_sha256(path)
        return entry.has_changed(current_hash)

    def get_chunk_ids_for_path(self, path: Path | str) -> list[str]:
        """Get all chunk IDs for a source file.

        Args:
            path: Path to the source file.

        Returns:
            List of chunk IDs, empty if not found.
        """
        entry = self.get_entry(path)
        return entry.chunk_ids if entry else []

    def get_sources_by_collection(self, collection: str) -> list[SourceEntry]:
        """Get all sources in a collection.

        Args:
            collection: Collection name.

        Returns:
            List of SourceEntry objects in the collection.
        """
        return [
            entry for entry in self.sources.values() if entry.collection == collection
        ]

    def save(self, path: Path) -> None:
        """Save manifest to a JSON file.

        Args:
            path: Path to save the manifest.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load manifest from a JSON file.

        Args:
            path: Path to the manifest file.

        Returns:
            Manifest instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the JSON is invalid.
        """
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def load_or_create(cls, path: Path) -> Manifest:
        """Load manifest from file or create new if not found.

        Args:
            path: Path to the manifest file.

        Returns:
            Manifest instance.
        """
        if path.exists():
            return cls.load(path)
        return cls()

    def stats(self) -> dict[str, Any]:
        """Get statistics about the manifest.

        Returns:
            Dictionary with source and chunk counts.
        """
        collections: dict[str, int] = {}
        total_chunks = 0

        for entry in self.sources.values():
            collections[entry.collection] = collections.get(entry.collection, 0) + 1
            total_chunks += entry.chunk_count

        return {
            "total_sources": len(self.sources),
            "total_chunks": total_chunks,
            "collections": collections,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
