"""Tests for the manifest repair script."""

from __future__ import annotations

from pathlib import Path

from grounded_code_mcp.manifest import Manifest, SourceEntry

# Import the function under test — will fail until we create the module.
from grounded_code_mcp.manifest_repair import build_repair_entries


class TestBuildRepairEntries:
    """Tests for build_repair_entries — pure logic that maps disk files + Qdrant
    chunk data into SourceEntry objects for manifest repair."""

    def test_builds_entry_for_file_with_chunks(self, temp_dir: Path) -> None:
        """A file on disk with matching Qdrant chunks produces a complete SourceEntry."""
        # Arrange
        src_file = temp_dir / "sources" / "dotnet" / "some-book.pdf"
        src_file.parent.mkdir(parents=True)
        src_file.write_bytes(b"fake pdf content")

        sources_dir = temp_dir / "sources"
        qdrant_chunks: dict[str, list[str]] = {
            "dotnet/some-book.pdf": ["chunk-1", "chunk-2", "chunk-3"],
        }

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert
        assert len(entries) == 1
        entry = entries[0]
        assert entry.path == "dotnet/some-book.pdf"
        assert entry.collection == "grounded_dotnet"
        assert entry.chunk_count == 3
        assert entry.chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
        assert entry.sha256 != ""
        assert entry.file_type == "pdf"

    def test_builds_entry_for_file_without_chunks(self, temp_dir: Path) -> None:
        """A file on disk with zero Qdrant chunks still gets an entry (like empty docs)."""
        # Arrange
        src_file = temp_dir / "sources" / "dotnet" / "empty-doc.md"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("")

        sources_dir = temp_dir / "sources"
        qdrant_chunks: dict[str, list[str]] = {}  # No chunks for this file

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert
        assert len(entries) == 1
        entry = entries[0]
        assert entry.path == "dotnet/empty-doc.md"
        assert entry.chunk_count == 0
        assert entry.chunk_ids == []

    def test_handles_multiple_files(self, temp_dir: Path) -> None:
        """Multiple files on disk each get their own entry."""
        # Arrange
        sources_dir = temp_dir / "sources"
        dotnet_dir = sources_dir / "dotnet"
        dotnet_dir.mkdir(parents=True)

        (dotnet_dir / "book-a.pdf").write_bytes(b"book a")
        (dotnet_dir / "book-b.pdf").write_bytes(b"book b")

        qdrant_chunks: dict[str, list[str]] = {
            "dotnet/book-a.pdf": ["a1", "a2"],
            "dotnet/book-b.pdf": ["b1"],
        }

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert
        assert len(entries) == 2
        paths = {e.path for e in entries}
        assert paths == {"dotnet/book-a.pdf", "dotnet/book-b.pdf"}

    def test_handles_nested_subdirectory(self, temp_dir: Path) -> None:
        """Files in nested subdirectories are handled correctly."""
        # Arrange
        sources_dir = temp_dir / "sources"
        nested = sources_dir / "dotnet" / "subdir"
        nested.mkdir(parents=True)
        (nested / "deep.md").write_text("# Deep file")

        qdrant_chunks: dict[str, list[str]] = {
            "dotnet/subdir/deep.md": ["d1"],
        }

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert
        assert len(entries) == 1
        assert entries[0].path == "dotnet/subdir/deep.md"

    def test_skips_non_ingestable_extensions(self, temp_dir: Path) -> None:
        """Files with unsupported extensions are skipped."""
        # Arrange
        sources_dir = temp_dir / "sources"
        dotnet_dir = sources_dir / "dotnet"
        dotnet_dir.mkdir(parents=True)

        (dotnet_dir / "notes.pdf").write_bytes(b"pdf content")
        (dotnet_dir / "image.jpg").write_bytes(b"not ingestable")
        (dotnet_dir / "data.csv").write_text("a,b,c")

        qdrant_chunks: dict[str, list[str]] = {
            "dotnet/notes.pdf": ["c1"],
        }

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert — only the .pdf, not .jpg or .csv
        assert len(entries) == 1
        assert entries[0].path == "dotnet/notes.pdf"

    def test_empty_directory_returns_empty(self, temp_dir: Path) -> None:
        """An empty source directory produces no entries."""
        # Arrange
        sources_dir = temp_dir / "sources"
        (sources_dir / "dotnet").mkdir(parents=True)
        qdrant_chunks: dict[str, list[str]] = {}

        # Act
        entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Assert
        assert entries == []


class TestRepairManifestIntegration:
    """Integration test: build entries and merge them into a Manifest."""

    def test_entries_merge_into_manifest(self, temp_dir: Path) -> None:
        """Repair entries can be added to an existing manifest without data loss."""
        # Arrange: existing manifest with an internal entry
        manifest = Manifest()
        existing = SourceEntry(
            path="internal/doc.md",
            sha256="abc123",
            collection="grounded_internal",
            chunk_count=5,
            chunk_ids=["i1", "i2", "i3", "i4", "i5"],
        )
        manifest.add_entry(existing)

        # Build repair entries
        sources_dir = temp_dir / "sources"
        dotnet_dir = sources_dir / "dotnet"
        dotnet_dir.mkdir(parents=True)
        (dotnet_dir / "book.pdf").write_bytes(b"book content")

        qdrant_chunks: dict[str, list[str]] = {
            "dotnet/book.pdf": ["d1", "d2"],
        }
        repair_entries = build_repair_entries(
            sources_dir=sources_dir,
            sub_dir="dotnet",
            collection_name="grounded_dotnet",
            qdrant_chunks=qdrant_chunks,
        )

        # Act: merge repair entries into manifest
        for entry in repair_entries:
            manifest.add_entry(entry)

        # Assert: both the original and repaired entries exist
        assert manifest.get_entry("internal/doc.md") is not None
        assert manifest.get_entry("dotnet/book.pdf") is not None
        assert manifest.get_entry("dotnet/book.pdf").chunk_count == 2  # type: ignore[union-attr]
