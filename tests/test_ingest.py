"""Tests for the ingestion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grounded_code_mcp.config import (
    ChunkingSettings,
    KnowledgeBaseSettings,
    OllamaSettings,
    Settings,
    VectorStoreSettings,
)
from grounded_code_mcp.embeddings import EmbeddingResult
from grounded_code_mcp.ingest import (
    IngestionPipeline,
    IngestStats,
    ingest_documents,
)


class TestIngestStats:
    """Tests for IngestStats dataclass."""

    def test_success_when_no_failures(self) -> None:
        """Test success is True when no files failed."""
        stats = IngestStats(files_ingested=5, files_skipped=2, files_failed=0)
        assert stats.success is True

    def test_failure_when_files_failed(self) -> None:
        """Test success is False when files failed."""
        stats = IngestStats(files_ingested=5, files_skipped=2, files_failed=1)
        assert stats.success is False


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        """Create settings for testing."""
        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        return Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=sources_dir,
                data_dir=data_dir,
            ),
            ollama=OllamaSettings(
                model="test-model",
                embedding_dim=128,
            ),
            chunking=ChunkingSettings(
                text_chunk_size=100,
                text_chunk_max_size=200,
            ),
            vectorstore=VectorStoreSettings(provider="qdrant"),
        )

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedding client."""
        embedder = MagicMock()
        embedder.ensure_ready.return_value = None
        embedder.embed_many.side_effect = lambda texts, **_kw: [
            EmbeddingResult(text=t, embedding=[0.1] * 128, model="test") for t in texts
        ]
        return embedder

    @pytest.fixture
    def pipeline(self, settings: Settings, mock_embedder: MagicMock) -> IngestionPipeline:
        """Create a pipeline with mocked embedder."""
        return IngestionPipeline(settings, embedder=mock_embedder)

    def test_ingest_empty_directory(self, pipeline: IngestionPipeline, settings: Settings) -> None:
        """Test ingesting an empty directory."""
        stats = pipeline.ingest()

        assert stats.files_scanned == 0
        assert stats.files_ingested == 0
        assert stats.success is True

    def test_ingest_single_file(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test ingesting a single markdown file."""
        # Create a test file
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")

        stats = pipeline.ingest()

        assert stats.files_scanned == 1
        assert stats.files_ingested == 1
        assert stats.success is True

    def test_ingest_skips_unchanged(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test that unchanged files are skipped."""
        # Create a test file
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")

        # First ingest
        stats1 = pipeline.ingest()
        assert stats1.files_ingested == 1

        # Second ingest should skip
        stats2 = pipeline.ingest()
        assert stats2.files_ingested == 0
        assert stats2.files_skipped == 1

    def test_ingest_force_reprocess(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test force flag causes reprocessing."""
        # Create a test file
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")

        # First ingest
        pipeline.ingest()

        # Force re-ingest
        stats = pipeline.ingest(force=True)
        assert stats.files_ingested == 1

    def test_ingest_nonexistent_path(self, pipeline: IngestionPipeline, settings: Settings) -> None:
        """Test ingesting nonexistent path returns empty stats."""
        nonexistent = settings.knowledge_base.sources_dir / "nonexistent"
        stats = pipeline.ingest(nonexistent)

        assert stats.files_scanned == 0
        assert stats.success is True

    def test_ingest_handles_embedder_error(self, settings: Settings, temp_dir: Path) -> None:
        """Test handling embedder connection errors."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "http://localhost:11434", "Connection refused"
        )

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)
        stats = pipeline.ingest()

        assert stats.success is False
        assert stats.files_failed > 0
        assert len(stats.errors) > 0

    def test_remove_source(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test removing a source."""
        # Create and ingest a file
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nContent.")

        pipeline.ingest()

        # Remove the source
        result = pipeline.remove_source(Path("test.md"))

        assert result is True
        assert pipeline.manifest.get_entry("test.md") is None

    def test_remove_nonexistent_source(self, pipeline: IngestionPipeline) -> None:
        """Test removing a source that doesn't exist."""
        result = pipeline.remove_source("nonexistent.md")
        assert result is False

    def test_chunks_created_reports_current_run_only(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test that chunks_created counts only chunks from the current run, not cumulative."""
        # Arrange: create two files
        file_a = settings.knowledge_base.sources_dir / "file_a.md"
        file_a.write_text("# File A\n\nContent for file A.")
        file_b = settings.knowledge_base.sources_dir / "file_b.md"
        file_b.write_text("# File B\n\nContent for file B.")

        # Act: ingest each file separately
        stats_a = pipeline.ingest(file_a)
        chunks_a = stats_a.chunks_created
        assert chunks_a > 0

        stats_b = pipeline.ingest(file_b)
        chunks_b = stats_b.chunks_created
        assert chunks_b > 0

        # Assert: sum of per-run counts should equal total manifest chunks.
        # If chunks_created is cumulative (the bug), chunks_b would include
        # file_a's chunks, making the sum too large.
        total_manifest_chunks = sum(e.chunk_count for e in pipeline.manifest.sources.values())
        assert chunks_a + chunks_b == total_manifest_chunks

    def test_ingest_empty_document_is_tracked_in_manifest(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
    ) -> None:
        """Test that empty documents are tracked in the manifest to prevent re-scanning."""
        # Arrange: a file that parses to empty content
        empty_file = settings.knowledge_base.sources_dir / "empty.md"
        empty_file.write_text("")

        # Act
        stats = pipeline.ingest()

        # Assert: counted as skipped (not ingested), but tracked in manifest
        assert stats.files_scanned == 1
        assert stats.files_skipped == 1
        assert stats.files_ingested == 0
        assert pipeline.manifest.get_entry(Path("empty.md")) is not None

    def test_ingest_empty_document_skipped_on_second_run(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
    ) -> None:
        """Test that a tracked empty document is skipped on subsequent runs."""
        # Arrange
        empty_file = settings.knowledge_base.sources_dir / "empty.md"
        empty_file.write_text("")

        # First run tracks it
        pipeline.ingest()

        # Act: second run
        stats = pipeline.ingest()

        # Assert: skipped via manifest hash check, not re-evaluated
        assert stats.files_skipped == 1
        assert stats.files_ingested == 0

    def test_user_provided_collection_gets_prefix(
        self,
        pipeline: IngestionPipeline,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test that user-provided collection name gets the configured prefix."""
        # Arrange
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")

        # Act: ingest with explicit collection name
        pipeline.ingest(test_file, collection="mycoll")

        # Assert: manifest entry should have prefixed collection name
        entry = pipeline.manifest.get_entry(Path("test.md"))
        assert entry is not None
        expected_prefix = settings.vectorstore.collection_prefix
        assert entry.collection == f"{expected_prefix}mycoll"

    def test_manifest_saved_incrementally_after_each_file(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test that the manifest is saved to disk after each file, not only at the end.

        If the process is killed mid-batch (OOM, SIGKILL, native crash), files
        already ingested must be recorded on disk. This prevents the "data in
        Qdrant but nothing in manifest" scenario that caused dotnet/automation
        collections to appear perpetually untracked.
        """
        from grounded_code_mcp.manifest import Manifest

        # Arrange: create two files in source dir
        file_a = settings.knowledge_base.sources_dir / "file_a.md"
        file_a.write_text("# File A\n\nContent for file A.")
        file_b = settings.knowledge_base.sources_dir / "file_b.md"
        file_b.write_text("# File B\n\nContent for file B.")

        # Track how many times the manifest exists on disk after embed_many calls.
        # embed_many is called once per file, so after the first call completes
        # and _process_file returns, the manifest should be on disk.
        disk_snapshots: list[int] = []
        manifest_path = settings.knowledge_base.manifest_path
        original_embed_many = mock_embedder.embed_many.side_effect

        def embed_and_snapshot(texts: list[str], **_kw: object) -> list[EmbeddingResult]:
            """After each embed call, the previous file's save should be on disk."""
            if manifest_path.exists():
                on_disk = Manifest.load(manifest_path)
                disk_snapshots.append(len(on_disk.sources))
            else:
                disk_snapshots.append(0)
            return original_embed_many(texts, **_kw)

        mock_embedder.embed_many.side_effect = embed_and_snapshot
        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        # Act
        pipeline.ingest()

        # Assert: by the time we process file_b (second embed_many call),
        # file_a should already be on disk — so disk_snapshots[1] >= 1
        assert len(disk_snapshots) == 2, f"Expected 2 embed calls, got {len(disk_snapshots)}"
        assert disk_snapshots[1] >= 1, (
            f"After first file was ingested, expected ≥1 entry on disk "
            f"before second file started, but found {disk_snapshots[1]}. "
            f"Manifest must be saved incrementally after each file."
        )


class TestRebuildCollection:
    """Tests for IngestionPipeline.rebuild_collection."""

    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        """Create settings for testing."""
        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        return Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=sources_dir,
                data_dir=data_dir,
            ),
            ollama=OllamaSettings(
                model="test-model",
                embedding_dim=128,
            ),
            chunking=ChunkingSettings(
                text_chunk_size=100,
                text_chunk_max_size=200,
            ),
            vectorstore=VectorStoreSettings(provider="qdrant"),
        )

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedding client."""
        embedder = MagicMock()
        embedder.ensure_ready.return_value = None
        embedder.embed_many.side_effect = lambda texts, **_kw: [
            EmbeddingResult(text=t, embedding=[0.1] * 128, model="test") for t in texts
        ]
        return embedder

    def test_rebuild_collection(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test rebuilding an entire collection."""
        # Create a test file and ingest it
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)
        pipeline.ingest()

        # Rebuild the collection
        collections = pipeline.store.list_collections()
        assert len(collections) > 0

        stats = pipeline.rebuild_collection(collections[0])
        assert stats.success is True

    def test_rebuild_empty_collection(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
    ) -> None:
        """Test rebuilding a collection with no matching sources."""
        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        stats = pipeline.rebuild_collection("nonexistent_collection")
        assert stats.success is True
        assert stats.files_scanned == 0


class TestIngestDocuments:
    """Tests for ingest_documents convenience function."""

    def test_ingest_documents(self, temp_dir: Path) -> None:
        """Test the convenience function."""
        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=sources_dir,
                data_dir=data_dir,
            ),
            ollama=OllamaSettings(
                model="test-model",
                embedding_dim=128,
            ),
            vectorstore=VectorStoreSettings(provider="qdrant"),
        )

        # Create a test file
        (sources_dir / "test.md").write_text("# Test\n\nContent.")

        # Mock the embedder
        with patch("grounded_code_mcp.ingest.EmbeddingClient") as mock_client_class:
            mock_embedder = MagicMock()
            mock_embedder.ensure_ready.return_value = None
            mock_embedder.embed_many.side_effect = lambda texts, **_kw: [
                EmbeddingResult(text=t, embedding=[0.1] * 128, model="test") for t in texts
            ]
            mock_client_class.from_settings.return_value = mock_embedder

            stats = ingest_documents(settings)

        assert stats.files_scanned == 1
