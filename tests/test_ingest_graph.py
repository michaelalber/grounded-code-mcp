"""Tests for graph rebuild integration in IngestionPipeline.

Graph rebuild is triggered only when force=True.  These tests patch
graph.graph_builder.build and graph.graph_store.GraphStore so no
real NetworkX/disk I/O is required.
"""

from __future__ import annotations

import logging
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
from grounded_code_mcp.ingest import IngestionPipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(temp_dir: Path) -> Settings:
    sources_dir = temp_dir / "sources"
    sources_dir.mkdir()
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    return Settings(
        knowledge_base=KnowledgeBaseSettings(sources_dir=sources_dir, data_dir=data_dir),
        ollama=OllamaSettings(model="test-model", embedding_dim=128),
        chunking=ChunkingSettings(text_chunk_size=100, text_chunk_max_size=200),
        vectorstore=VectorStoreSettings(provider="qdrant"),
    )


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.ensure_ready.return_value = None
    embedder.embed_many.side_effect = lambda texts, **_kw: [
        EmbeddingResult(text=t, embedding=[0.1] * 128, model="test") for t in texts
    ]
    return embedder


@pytest.fixture()
def pipeline(settings: Settings, mock_embedder: MagicMock) -> IngestionPipeline:
    return IngestionPipeline(settings, embedder=mock_embedder)


# ---------------------------------------------------------------------------
# Single-source force: RELATIONSHIPS.md present
# ---------------------------------------------------------------------------


class TestForceSingleSourceWithRelationships:
    def test_build_called_with_relationships_file(
        self, settings: Settings, mock_embedder: MagicMock, temp_dir: Path
    ) -> None:
        """force=True + directory that has RELATIONSHIPS.md → build() called once."""
        source_dir = settings.knowledge_base.sources_dir / "my-source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")
        rel_file = source_dir / "RELATIONSHIPS.md"
        rel_file.write_text('"Concept A" → enables → "Concept B"\n')

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore") as mock_store_cls,
            patch("graph.graph_builder.build") as mock_build,
        ):
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            pipeline.ingest(source_dir, force=True)

        mock_build.assert_called_once()
        args = mock_build.call_args
        called_path = args[0][0]
        assert called_path == rel_file

    def test_graph_store_loaded_before_build(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """GraphStore.load() is called before build()."""
        source_dir = settings.knowledge_base.sources_dir / "src"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")
        (source_dir / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)
        call_order: list[str] = []

        with (
            patch("graph.graph_store.GraphStore") as mock_store_cls,
            patch("graph.graph_builder.build") as mock_build,
        ):
            mock_store = MagicMock()
            mock_store.load.side_effect = lambda: call_order.append("load")
            mock_store_cls.return_value = mock_store
            mock_build.side_effect = lambda *a, **kw: call_order.append("build")

            pipeline.ingest(source_dir, force=True)

        assert call_order.index("load") < call_order.index("build")

    def test_info_logged_after_build(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After successful graph rebuild, an INFO message names the source slug."""
        source_dir = settings.knowledge_base.sources_dir / "tidy-first"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")
        (source_dir / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build"),
            caplog.at_level(logging.INFO, logger="grounded_code_mcp.ingest"),
        ):
            pipeline.ingest(source_dir, force=True)

        assert any("tidy-first" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Single-source force: RELATIONSHIPS.md absent
# ---------------------------------------------------------------------------


class TestForceSingleSourceWithoutRelationships:
    def test_build_not_called_when_no_relationships_file(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """force=True + directory with no RELATIONSHIPS.md → build() never called."""
        source_dir = settings.knowledge_base.sources_dir / "orphan-source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build") as mock_build,
        ):
            pipeline.ingest(source_dir, force=True)

        mock_build.assert_not_called()

    def test_warning_logged_when_no_relationships_file(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """force=True + no RELATIONSHIPS.md → WARNING logged with source slug."""
        source_dir = settings.knowledge_base.sources_dir / "orphan-source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build"),
            caplog.at_level(logging.WARNING, logger="grounded_code_mcp.ingest"),
        ):
            pipeline.ingest(source_dir, force=True)

        assert any(
            "orphan-source" in r.message and "RELATIONSHIPS.md" in r.message for r in caplog.records
        )

    def test_ingest_still_succeeds_when_no_relationships_file(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """Missing RELATIONSHIPS.md must not cause ingest to fail."""
        source_dir = settings.knowledge_base.sources_dir / "orphan-source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build"),
        ):
            stats = pipeline.ingest(source_dir, force=True)

        assert stats.success is True
        assert stats.files_ingested == 1


# ---------------------------------------------------------------------------
# Global force (no specific path)
# ---------------------------------------------------------------------------


class TestForceGlobalRebuild:
    def test_build_called_with_sources_dir(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """force=True + no path → build() called with sources_dir."""
        (settings.knowledge_base.sources_dir / "a.md").write_text("# A\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build") as mock_build,
        ):
            pipeline.ingest(force=True)

        mock_build.assert_called_once()
        called_path = mock_build.call_args[0][0]
        assert called_path == settings.knowledge_base.sources_dir

    def test_info_logged_with_file_count(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Global rebuild logs INFO with count of RELATIONSHIPS.md files processed."""
        sources_dir = settings.knowledge_base.sources_dir
        sub = sources_dir / "collection-a"
        sub.mkdir()
        (sub / "doc.md").write_text("# A\n\nContent.")
        (sub / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        from graph.graph_builder import BuildStats

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build", return_value=BuildStats(files_processed=1)) as _,
            caplog.at_level(logging.INFO, logger="grounded_code_mcp.ingest"),
        ):
            pipeline.ingest(force=True)

        assert any("1" in r.message and "RELATIONSHIPS" in r.message for r in caplog.records)

    def test_skipped_count_in_log(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Global rebuild log includes count of source dirs that had no RELATIONSHIPS.md."""
        sources_dir = settings.knowledge_base.sources_dir
        # One dir WITH relationships
        with_rel = sources_dir / "with-rel"
        with_rel.mkdir()
        (with_rel / "doc.md").write_text("# With\n\nContent.")
        (with_rel / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')
        # One dir WITHOUT relationships
        without_rel = sources_dir / "without-rel"
        without_rel.mkdir()
        (without_rel / "doc.md").write_text("# Without\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        from graph.graph_builder import BuildStats

        with (
            patch("graph.graph_store.GraphStore"),
            patch("graph.graph_builder.build", return_value=BuildStats(files_processed=1)),
            caplog.at_level(logging.INFO, logger="grounded_code_mcp.ingest"),
        ):
            pipeline.ingest(force=True)

        # The log must mention both N=1 (files) and M=1 (skipped)
        combined = " ".join(r.message for r in caplog.records)
        assert "1" in combined and "skipped" in combined.lower()


# ---------------------------------------------------------------------------
# No-force path: graph must never be touched
# ---------------------------------------------------------------------------


class TestNoForceDoesNotTouchGraph:
    def test_build_not_called_without_force(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """Normal ingest (force=False) must never call build()."""
        source_dir = settings.knowledge_base.sources_dir / "src"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")
        (source_dir / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with patch("graph.graph_builder.build") as mock_build:
            pipeline.ingest(source_dir, force=False)

        mock_build.assert_not_called()

    def test_graph_store_not_instantiated_without_force(
        self, settings: Settings, mock_embedder: MagicMock
    ) -> None:
        """Normal ingest must not create a GraphStore instance."""
        source_dir = settings.knowledge_base.sources_dir / "src"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Doc\n\nContent.")

        pipeline = IngestionPipeline(settings, embedder=mock_embedder)

        with patch("graph.graph_store.GraphStore") as mock_store_cls:
            pipeline.ingest(source_dir, force=False)

        mock_store_cls.assert_not_called()

    def test_existing_tests_unaffected(
        self, pipeline: IngestionPipeline, settings: Settings
    ) -> None:
        """Verify the no-force single-file path is completely unchanged."""
        test_file = settings.knowledge_base.sources_dir / "test.md"
        test_file.write_text("# Test Document\n\nSome content here.")
        stats = pipeline.ingest()
        assert stats.files_ingested == 1
        assert stats.success is True
