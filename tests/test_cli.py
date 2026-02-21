"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from grounded_code_mcp.__main__ import cli
from grounded_code_mcp.ingest import IngestStats


def test_cli_version() -> None:
    """Test that --version flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help() -> None:
    """Test that --help flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "grounded-code-mcp" in result.output


def test_ingest_command_exists() -> None:
    """Test that ingest command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "Ingest documents" in result.output


def test_status_command_exists() -> None:
    """Test that status command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output.lower()


def test_serve_command_exists() -> None:
    """Test that serve command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output


def test_search_command_exists() -> None:
    """Test that search command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "Search" in result.output


class TestIngestCommand:
    """Integration tests for the ingest CLI command."""

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.ingest.ingest_documents")
    def test_ingest_success(self, mock_ingest: MagicMock, mock_settings_cls: MagicMock) -> None:
        """Test successful ingestion output."""
        mock_ingest.return_value = IngestStats(
            files_scanned=5,
            files_ingested=3,
            files_skipped=2,
            files_failed=0,
            chunks_created=15,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest"])

        assert result.exit_code == 0
        assert "Starting ingestion" in result.output
        assert "Ingested 3 files" in result.output
        assert "Scanned: 5" in result.output
        assert "Skipped: 2" in result.output
        assert "Chunks created: 15" in result.output
        # Verify settings from Settings.load() are passed to ingest_documents
        mock_ingest.assert_called_once_with(
            mock_settings_cls.load.return_value,
            path=None,
            collection=None,
            force=False,
        )

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.ingest.ingest_documents")
    def test_ingest_with_errors(
        self, mock_ingest: MagicMock, _mock_settings_cls: MagicMock
    ) -> None:
        """Test ingestion with errors shows error output."""
        mock_ingest.return_value = IngestStats(
            files_scanned=3,
            files_ingested=1,
            files_skipped=0,
            files_failed=2,
            errors=["file1.pdf: Parse error", "file2.doc: Unsupported format"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest"])

        assert result.exit_code == 0
        assert "Ingestion completed with errors" in result.output
        assert "Files failed: 2" in result.output
        assert "Parse error" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.ingest.ingest_documents")
    def test_ingest_with_force_flag(
        self, mock_ingest: MagicMock, _mock_settings_cls: MagicMock
    ) -> None:
        """Test that --force flag is passed through."""
        mock_ingest.return_value = IngestStats(files_ingested=1)

        runner = CliRunner()
        runner.invoke(cli, ["ingest", "--force"])

        _, kwargs = mock_ingest.call_args
        assert kwargs["force"] is True

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.ingest.ingest_documents")
    def test_ingest_with_collection(
        self, mock_ingest: MagicMock, _mock_settings_cls: MagicMock
    ) -> None:
        """Test that --collection is passed through."""
        mock_ingest.return_value = IngestStats(files_ingested=1)

        runner = CliRunner()
        runner.invoke(cli, ["ingest", "--collection", "python"])

        _, kwargs = mock_ingest.call_args
        assert kwargs["collection"] == "python"


class TestStatusCommand:
    """Integration tests for the status CLI command."""

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_status_healthy(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test status with healthy Ollama and no manifest."""
        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir / "sources"
        mock_settings.knowledge_base.data_dir = temp_dir / "data"
        mock_settings.knowledge_base.manifest_path = temp_dir / "manifest.json"
        mock_settings.vectorstore.provider = "qdrant"
        mock_settings.ollama.model = "snowflake-arctic-embed2"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.health_check.return_value = {
            "healthy": True,
            "model": "snowflake-arctic-embed2",
        }
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Knowledge Base Status" in result.output
        assert "Ollama is running" in result.output
        assert "No manifest found" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_status_unhealthy_ollama(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test status with unhealthy Ollama."""
        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir / "sources"
        mock_settings.knowledge_base.data_dir = temp_dir / "data"
        mock_settings.knowledge_base.manifest_path = temp_dir / "manifest.json"
        mock_settings.vectorstore.provider = "qdrant"
        mock_settings.ollama.model = "snowflake-arctic-embed2"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.health_check.return_value = {
            "healthy": False,
            "error": "Connection refused",
        }
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Ollama is not ready" in result.output
        assert "Connection refused" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.manifest.Manifest.load")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_status_with_manifest_and_collections(
        self,
        mock_create_store: MagicMock,
        mock_manifest_load: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test status showing manifest stats and vector store collections."""
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text("{}")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir / "sources"
        mock_settings.knowledge_base.data_dir = temp_dir / "data"
        mock_settings.knowledge_base.manifest_path = manifest_path
        mock_settings.vectorstore.provider = "qdrant"
        mock_settings.ollama.model = "snowflake-arctic-embed2"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.health_check.return_value = {"healthy": True, "model": "snowflake-arctic-embed2"}
        mock_from_settings.return_value = mock_embedder

        mock_manifest = MagicMock()
        mock_manifest.stats.return_value = {
            "total_sources": 5,
            "total_chunks": 42,
            "updated_at": "2026-01-15T10:00:00",
            "collections": {"grounded_python": 3, "grounded_patterns": 2},
        }
        mock_manifest_load.return_value = mock_manifest

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python", "grounded_patterns"]
        mock_store.collection_count.side_effect = [30, 12]
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Total sources: 5" in result.output
        assert "Total chunks: 42" in result.output
        assert "grounded_python" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_status_vector_store_error(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test status handles vector store errors gracefully."""
        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir / "sources"
        mock_settings.knowledge_base.data_dir = temp_dir / "data"
        mock_settings.knowledge_base.manifest_path = temp_dir / "manifest.json"
        mock_settings.vectorstore.provider = "qdrant"
        mock_settings.ollama.model = "snowflake-arctic-embed2"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.health_check.return_value = {"healthy": True, "model": "test"}
        mock_from_settings.return_value = mock_embedder

        mock_create_store.side_effect = RuntimeError("Store unavailable")

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Could not read vector store" in result.output


class TestServeCommand:
    """Integration tests for the serve CLI command."""

    @patch("grounded_code_mcp.server.run_server")
    def test_serve_starts(self, mock_run_server: MagicMock) -> None:
        """Test serve command launches the server."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve"])

        assert result.exit_code == 0
        assert "Starting MCP server" in result.output
        mock_run_server.assert_called_once_with(debug=False)

    @patch("grounded_code_mcp.server.run_server")
    def test_serve_debug_mode(self, mock_run_server: MagicMock) -> None:
        """Test serve --debug passes debug flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--debug"])

        assert result.exit_code == 0
        assert "Debug mode enabled" in result.output
        mock_run_server.assert_called_once_with(debug=True)


class TestSearchCommand:
    """Integration tests for the search CLI command."""

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_with_results(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search displaying results."""
        from grounded_code_mcp.embeddings import EmbeddingResult
        from grounded_code_mcp.vectorstore import SearchResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="test query", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [
            SearchResult(
                chunk_id="id1",
                content="This is a relevant document chunk.",
                score=0.92,
                metadata={
                    "source_path": "docs/guide.md",
                    "heading_context": ["Guide", "Setup"],
                },
            ),
        ]
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test query"])

        assert result.exit_code == 0
        assert "Searching for: test query" in result.output
        assert "Result 1" in result.output
        assert "0.9200" in result.output
        assert "docs/guide.md" in result.output
        assert "Guide > Setup" in result.output
        assert "relevant document chunk" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_no_results(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search with no results."""
        from grounded_code_mcp.embeddings import EmbeddingResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="obscure", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "obscure query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_no_collections(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search when no collections exist."""
        from grounded_code_mcp.embeddings import EmbeddingResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="query", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test query"])

        assert result.exit_code == 0
        assert "No collections found" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    def test_search_ollama_not_ready(
        self,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search when Ollama is not available."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_settings = MagicMock()
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "http://localhost:11434", "Connection refused"
        )
        mock_from_settings.return_value = mock_embedder

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test query"])

        assert result.exit_code == 0
        assert "Cannot connect to Ollama" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_with_collection_filter(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search with --collection flag."""
        from grounded_code_mcp.embeddings import EmbeddingResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="query", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        runner.invoke(cli, ["search", "test query", "--collection", "python"])

        # Should search only the specified collection
        mock_store.search.assert_called_once()
        call_args = mock_store.search.call_args
        assert call_args[0][0] == "grounded_python"

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_with_long_content_truncated(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test that long search results are truncated in display."""
        from grounded_code_mcp.embeddings import EmbeddingResult
        from grounded_code_mcp.vectorstore import SearchResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="query", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        long_content = "A" * 600
        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [
            SearchResult(chunk_id="id1", content=long_content, score=0.9),
        ]
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "query"])

        assert result.exit_code == 0
        assert "..." in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_handles_collection_error(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Test search handles errors from individual collection searches."""
        from grounded_code_mcp.embeddings import EmbeddingResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="query", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_broken"]
        mock_store.search.side_effect = RuntimeError("Collection corrupted")
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "query"])

        assert result.exit_code == 0
        assert "Error searching" in result.output
