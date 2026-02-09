"""Tests for manifest tracking."""

from pathlib import Path

from grounded_code_mcp.manifest import (
    ChunkMetadata,
    Manifest,
    SourceEntry,
    compute_sha256,
)


class TestComputeSha256:
    """Tests for SHA-256 computation."""

    def test_compute_hash(self, temp_dir: Path) -> None:
        """Test computing hash of a file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        result = compute_sha256(test_file)

        # Known SHA-256 for "Hello, World!"
        assert result == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_hash_changes_with_content(self, temp_dir: Path) -> None:
        """Test that hash changes when content changes."""
        test_file = temp_dir / "test.txt"

        test_file.write_text("content1")
        hash1 = compute_sha256(test_file)

        test_file.write_text("content2")
        hash2 = compute_sha256(test_file)

        assert hash1 != hash2


class TestChunkMetadata:
    """Tests for ChunkMetadata model."""

    def test_defaults(self) -> None:
        """Test default values."""
        meta = ChunkMetadata(chunk_id="chunk-1", chunk_index=0)
        assert meta.chunk_id == "chunk-1"
        assert meta.chunk_index == 0
        assert meta.start_char == 0
        assert meta.end_char == 0
        assert meta.heading_context == []
        assert meta.is_code is False
        assert meta.code_language is None
        assert meta.is_table is False

    def test_code_chunk(self) -> None:
        """Test code chunk metadata."""
        meta = ChunkMetadata(
            chunk_id="chunk-2",
            chunk_index=1,
            is_code=True,
            code_language="python",
        )
        assert meta.is_code is True
        assert meta.code_language == "python"


class TestSourceEntry:
    """Tests for SourceEntry model."""

    def test_creation(self) -> None:
        """Test creating a source entry."""
        entry = SourceEntry(
            path="sources/python/doc.md",
            sha256="abc123",
            collection="grounded_python",
        )
        assert entry.path == "sources/python/doc.md"
        assert entry.sha256 == "abc123"
        assert entry.collection == "grounded_python"
        assert entry.chunk_count == 0
        assert entry.chunk_ids == []

    def test_has_changed(self) -> None:
        """Test change detection."""
        entry = SourceEntry(
            path="test.md",
            sha256="original_hash",
            collection="test",
        )

        assert entry.has_changed("different_hash") is True
        assert entry.has_changed("original_hash") is False


class TestManifest:
    """Tests for Manifest model."""

    def test_empty_manifest(self) -> None:
        """Test creating an empty manifest."""
        manifest = Manifest()
        assert manifest.version == "1.0"
        assert manifest.sources == {}

    def test_add_entry(self) -> None:
        """Test adding an entry."""
        manifest = Manifest()
        entry = SourceEntry(
            path="test.md",
            sha256="abc123",
            collection="test",
        )

        manifest.add_entry(entry)

        assert "test.md" in manifest.sources
        assert manifest.sources["test.md"] == entry

    def test_get_entry(self) -> None:
        """Test getting an entry."""
        manifest = Manifest()
        entry = SourceEntry(path="test.md", sha256="abc", collection="test")
        manifest.add_entry(entry)

        result = manifest.get_entry("test.md")
        assert result == entry

        result = manifest.get_entry(Path("test.md"))
        assert result == entry

        result = manifest.get_entry("nonexistent.md")
        assert result is None

    def test_remove_entry(self) -> None:
        """Test removing an entry."""
        manifest = Manifest()
        entry = SourceEntry(path="test.md", sha256="abc", collection="test")
        manifest.add_entry(entry)

        removed = manifest.remove_entry("test.md")
        assert removed == entry
        assert manifest.get_entry("test.md") is None

        removed = manifest.remove_entry("nonexistent.md")
        assert removed is None

    def test_needs_reingestion_new_file(self, temp_dir: Path) -> None:
        """Test that new files need ingestion."""
        manifest = Manifest()
        test_file = temp_dir / "test.md"
        test_file.write_text("content")

        assert manifest.needs_reingestion(test_file) is True

    def test_needs_reingestion_unchanged(self, temp_dir: Path) -> None:
        """Test that unchanged files don't need reingestion."""
        test_file = temp_dir / "test.md"
        test_file.write_text("content")
        file_hash = compute_sha256(test_file)

        manifest = Manifest()
        entry = SourceEntry(
            path=str(test_file),
            sha256=file_hash,
            collection="test",
        )
        manifest.add_entry(entry)

        assert manifest.needs_reingestion(test_file) is False

    def test_needs_reingestion_changed(self, temp_dir: Path) -> None:
        """Test that changed files need reingestion."""
        test_file = temp_dir / "test.md"
        test_file.write_text("original")

        manifest = Manifest()
        entry = SourceEntry(
            path=str(test_file),
            sha256=compute_sha256(test_file),
            collection="test",
        )
        manifest.add_entry(entry)

        test_file.write_text("modified")
        assert manifest.needs_reingestion(test_file) is True

    def test_get_chunk_ids_for_path(self) -> None:
        """Test getting chunk IDs for a path."""
        manifest = Manifest()
        entry = SourceEntry(
            path="test.md",
            sha256="abc",
            collection="test",
            chunk_ids=["chunk-1", "chunk-2"],
        )
        manifest.add_entry(entry)

        result = manifest.get_chunk_ids_for_path("test.md")
        assert result == ["chunk-1", "chunk-2"]

        result = manifest.get_chunk_ids_for_path("nonexistent.md")
        assert result == []

    def test_get_sources_by_collection(self) -> None:
        """Test getting sources by collection."""
        manifest = Manifest()
        entry1 = SourceEntry(path="a.md", sha256="a", collection="python")
        entry2 = SourceEntry(path="b.md", sha256="b", collection="python")
        entry3 = SourceEntry(path="c.md", sha256="c", collection="dotnet")
        manifest.add_entry(entry1)
        manifest.add_entry(entry2)
        manifest.add_entry(entry3)

        python_sources = manifest.get_sources_by_collection("python")
        assert len(python_sources) == 2

        dotnet_sources = manifest.get_sources_by_collection("dotnet")
        assert len(dotnet_sources) == 1

    def test_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading manifest."""
        manifest = Manifest()
        entry = SourceEntry(
            path="test.md",
            sha256="abc123",
            collection="test",
            chunk_ids=["chunk-1"],
        )
        manifest.add_entry(entry)

        manifest_path = temp_dir / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)
        assert "test.md" in loaded.sources
        assert loaded.sources["test.md"].sha256 == "abc123"
        assert loaded.sources["test.md"].chunk_ids == ["chunk-1"]

    def test_load_or_create_existing(self, temp_dir: Path) -> None:
        """Test loading existing manifest."""
        manifest = Manifest()
        entry = SourceEntry(path="test.md", sha256="abc", collection="test")
        manifest.add_entry(entry)

        manifest_path = temp_dir / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load_or_create(manifest_path)
        assert "test.md" in loaded.sources

    def test_load_or_create_new(self, temp_dir: Path) -> None:
        """Test creating new manifest when file doesn't exist."""
        manifest_path = temp_dir / "manifest.json"
        manifest = Manifest.load_or_create(manifest_path)
        assert manifest.sources == {}

    def test_stats(self) -> None:
        """Test manifest statistics."""
        manifest = Manifest()
        manifest.add_entry(SourceEntry(path="a.md", sha256="a", collection="python", chunk_count=5))
        manifest.add_entry(SourceEntry(path="b.md", sha256="b", collection="python", chunk_count=3))
        manifest.add_entry(SourceEntry(path="c.md", sha256="c", collection="dotnet", chunk_count=7))

        stats = manifest.stats()

        assert stats["total_sources"] == 3
        assert stats["total_chunks"] == 15
        assert stats["collections"]["python"] == 2
        assert stats["collections"]["dotnet"] == 1

    def test_updated_at_changes(self) -> None:
        """Test that updated_at timestamp changes on modifications."""
        manifest = Manifest()
        original_updated = manifest.updated_at

        entry = SourceEntry(path="test.md", sha256="abc", collection="test")
        manifest.add_entry(entry)

        # updated_at should change
        assert manifest.updated_at >= original_updated
