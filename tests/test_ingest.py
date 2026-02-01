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
        embedder.embed_many.return_value = [
            EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
        ]
        return embedder

    @pytest.fixture
    def pipeline(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> IngestionPipeline:
        """Create a pipeline with mocked embedder."""
        return IngestionPipeline(settings, embedder=mock_embedder)

    def test_ingest_empty_directory(
        self, pipeline: IngestionPipeline, settings: Settings
    ) -> None:
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

        # Mock embedder to return right number of embeddings
        mock_embedder.embed_many.return_value = [
            EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
        ]

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
        mock_embedder.embed_many.return_value = [
            EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
        ]
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
        mock_embedder.embed_many.return_value = [
            EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
        ]
        pipeline.ingest()

        # Force re-ingest
        stats = pipeline.ingest(force=True)
        assert stats.files_ingested == 1

    def test_ingest_nonexistent_path(
        self, pipeline: IngestionPipeline, settings: Settings
    ) -> None:
        """Test ingesting nonexistent path returns empty stats."""
        nonexistent = settings.knowledge_base.sources_dir / "nonexistent"
        stats = pipeline.ingest(nonexistent)

        assert stats.files_scanned == 0
        assert stats.success is True

    def test_ingest_handles_embedder_error(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        """Test handling embedder connection errors."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "http://localhost:11434", "Connection refused"
        )

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)
        stats = pipeline.ingest()

        assert stats.success is False
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

        mock_embedder.embed_many.return_value = [
            EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
        ]
        pipeline.ingest()

        # Remove the source
        result = pipeline.remove_source(Path("test.md"))

        assert result is True
        assert pipeline.manifest.get_entry("test.md") is None

    def test_remove_nonexistent_source(
        self, pipeline: IngestionPipeline
    ) -> None:
        """Test removing a source that doesn't exist."""
        result = pipeline.remove_source("nonexistent.md")
        assert result is False


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
        with patch(
            "grounded_code_mcp.ingest.EmbeddingClient"
        ) as mock_client_class:
            mock_embedder = MagicMock()
            mock_embedder.ensure_ready.return_value = None
            mock_embedder.embed_many.return_value = [
                EmbeddingResult(text="test", embedding=[0.1] * 128, model="test")
            ]
            mock_client_class.from_settings.return_value = mock_embedder

            stats = ingest_documents(settings)

        assert stats.files_scanned == 1
