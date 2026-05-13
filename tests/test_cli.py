"""Tests for CLI commands."""

import json as _json
from importlib.metadata import version
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from grounded_code_mcp.__main__ import cli
from grounded_code_mcp.ingest import IngestStats
from grounded_code_mcp.parser import ParsedDocument


def test_cli_version() -> None:
    """Test that --version flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert version("grounded-code-mcp") in result.output


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
        call_kwargs = mock_ingest.call_args.kwargs
        mock_ingest.assert_called_once_with(
            mock_settings_cls.load.return_value,
            path=None,
            collection=None,
            force=False,
            progress_callback=call_kwargs["progress_callback"],
        )
        assert callable(call_kwargs["progress_callback"])

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
        mock_embedder.health_check.return_value = {
            "healthy": True,
            "model": "snowflake-arctic-embed2",
        }
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
        mock_run_server.assert_called_once_with(
            debug=False, transport=None, host="127.0.0.1", port=8080
        )

    @patch("grounded_code_mcp.server.run_server")
    def test_serve_debug_mode(self, mock_run_server: MagicMock) -> None:
        """Test serve --debug passes debug flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--debug"])

        assert result.exit_code == 0
        assert "Debug mode enabled" in result.output
        mock_run_server.assert_called_once_with(
            debug=True, transport=None, host="127.0.0.1", port=8080
        )


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


class TestConvertCommand:
    """Tests for the convert CLI command."""

    def test_convert_command_exists(self) -> None:
        """convert command is registered and has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "convert" in result.output.lower()

    def test_convert_dry_run_prints_files_without_writing(self, temp_dir: Path) -> None:
        """--dry-run lists files that would be converted without writing sidecars."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "book.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser"),
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(temp_dir), "--dry-run"])

        assert result.exit_code == 0
        assert "Would convert" in result.output
        assert not (temp_dir / "book.pdf.md").exists()

    def test_convert_skips_existing_sidecar(self, temp_dir: Path) -> None:
        """Files with an existing sidecar are counted as skipped without --force."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        sidecar = temp_dir / "doc.pdf.md"
        sidecar.write_text("# Existing")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser"),
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(temp_dir)])

        assert result.exit_code == 0
        assert "1 skipped" in result.output
        assert sidecar.read_text() == "# Existing"

    def test_convert_force_overwrites_sidecar(self, temp_dir: Path) -> None:
        """--force re-converts and overwrites an existing sidecar."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        sidecar = temp_dir / "doc.pdf.md"
        sidecar.write_text("# Old")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ParsedDocument(
            path=pdf_file, content="# New content", file_type="pdf"
        )

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser") as mock_parser_cls,
        ):
            mock_cls.load.return_value = mock_settings
            mock_parser_cls.return_value = mock_parser_instance
            result = runner.invoke(cli, ["convert", str(pdf_file), "--force"])

        assert result.exit_code == 0
        assert sidecar.read_text() == "# New content"

    def test_convert_skips_plaintext_files(self, temp_dir: Path) -> None:
        """Plaintext files (.md, .rst, .txt) are never passed to Docling."""
        from grounded_code_mcp.config import DoclingSettings

        (temp_dir / "guide.md").write_text("# Guide")
        (temp_dir / "notes.rst").write_text("Notes")
        (temp_dir / "readme.txt").write_text("Readme")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser"),
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(temp_dir)])

        assert result.exit_code == 0
        assert "No files to convert" in result.output

    def test_convert_no_ocr_flag_disables_ocr(self, temp_dir: Path) -> None:
        """--no-ocr flag passes enable_ocr=False to DocumentParser."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "book.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ParsedDocument(
            path=pdf_file, content="# Content", file_type="pdf"
        )

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser") as mock_parser_cls,
        ):
            mock_cls.load.return_value = mock_settings
            mock_parser_cls.return_value = mock_parser_instance
            result = runner.invoke(cli, ["convert", str(pdf_file), "--no-ocr"])

        assert result.exit_code == 0
        mock_parser_cls.assert_called_once_with(
            enable_ocr=False, docling_settings=mock_settings.docling
        )

    def test_convert_respects_docling_settings_enable_ocr(self, temp_dir: Path) -> None:
        """DoclingSettings.enable_ocr=False is honoured without --no-ocr flag."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "book.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        docling_settings = DoclingSettings(enable_ocr=False)
        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = docling_settings

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ParsedDocument(
            path=pdf_file, content="# Content", file_type="pdf"
        )

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser") as mock_parser_cls,
        ):
            mock_cls.load.return_value = mock_settings
            mock_parser_cls.return_value = mock_parser_instance
            runner.invoke(cli, ["convert", str(pdf_file)])

        mock_parser_cls.assert_called_once_with(enable_ocr=False, docling_settings=docling_settings)

    def test_convert_no_ocr_overrides_config_enable_ocr(self, temp_dir: Path) -> None:
        """--no-ocr disables OCR even when DoclingSettings.enable_ocr is True."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "book.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        docling_settings = DoclingSettings(enable_ocr=True)
        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = docling_settings

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ParsedDocument(
            path=pdf_file, content="# Content", file_type="pdf"
        )

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser") as mock_parser_cls,
        ):
            mock_cls.load.return_value = mock_settings
            mock_parser_cls.return_value = mock_parser_instance
            runner.invoke(cli, ["convert", str(pdf_file), "--no-ocr"])

        mock_parser_cls.assert_called_once_with(enable_ocr=False, docling_settings=docling_settings)

    def test_convert_single_file_path_converts_that_file(self, temp_dir: Path) -> None:
        """Passing a single file path converts only that file."""
        from grounded_code_mcp.config import DoclingSettings

        pdf_file = temp_dir / "single.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ParsedDocument(
            path=pdf_file, content="# Single", file_type="pdf"
        )

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser") as mock_parser_cls,
        ):
            mock_cls.load.return_value = mock_settings
            mock_parser_cls.return_value = mock_parser_instance
            result = runner.invoke(cli, ["convert", str(pdf_file)])

        assert result.exit_code == 0
        assert "single.pdf" in result.output
        assert "1 converted" in result.output
        assert (temp_dir / "single.pdf.md").read_text() == "# Single"

    def test_convert_single_plaintext_file_reports_nothing_to_convert(self, temp_dir: Path) -> None:
        """Passing a single .md file path reports no files to convert."""
        from grounded_code_mcp.config import DoclingSettings

        md_file = temp_dir / "notes.md"
        md_file.write_text("# Notes")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.parser.DocumentParser"),
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(md_file)])

        assert result.exit_code == 0
        assert "No files to convert" in result.output

    def test_convert_directory_spawns_subprocess_per_file(self, temp_dir: Path) -> None:
        """Directory mode spawns an isolated subprocess for each file."""
        from grounded_code_mcp.config import DoclingSettings

        (temp_dir / "a.pdf").write_bytes(b"%PDF-1.4")
        (temp_dir / "b.pdf").write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.__main__.subprocess.run", return_value=mock_proc) as mock_sub,
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(temp_dir)])

        assert result.exit_code == 0
        assert mock_sub.call_count == 2
        assert "2 converted" in result.output

    def test_convert_directory_subprocess_crash_counted_as_failed(self, temp_dir: Path) -> None:
        """A subprocess with non-zero exit is counted as failed, not raised."""
        from grounded_code_mcp.config import DoclingSettings

        (temp_dir / "good.pdf").write_bytes(b"%PDF-1.4")
        (temp_dir / "bad.pdf").write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 1 if "bad.pdf" in cmd[-1] else 0
            proc.stderr = "Aborted (core dumped)" if "bad.pdf" in cmd[-1] else ""
            return proc

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.__main__.subprocess.run", side_effect=fake_run),
        ):
            mock_cls.load.return_value = mock_settings
            result = runner.invoke(cli, ["convert", str(temp_dir)])

        assert result.exit_code == 0
        assert "1 converted" in result.output
        assert "1 failed" in result.output
        assert "bad.pdf" in result.output

    def test_convert_directory_no_ocr_propagated_to_subprocess(self, temp_dir: Path) -> None:
        """--no-ocr is forwarded to each subprocess command in directory mode."""
        from grounded_code_mcp.config import DoclingSettings

        (temp_dir / "doc.pdf").write_bytes(b"%PDF-1.4")

        mock_settings = MagicMock()
        mock_settings.knowledge_base.sources_dir = temp_dir
        mock_settings.collections = {}
        mock_settings.docling = DoclingSettings()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        runner = CliRunner()
        with (
            patch("grounded_code_mcp.__main__.Settings") as mock_cls,
            patch("grounded_code_mcp.__main__.subprocess.run", return_value=mock_proc) as mock_sub,
        ):
            mock_cls.load.return_value = mock_settings
            runner.invoke(cli, ["convert", str(temp_dir), "--no-ocr"])

        cmd = mock_sub.call_args[0][0]
        assert "--no-ocr" in cmd


# ---------------------------------------------------------------------------
# Tests for --json flag on the existing `search` command
# ---------------------------------------------------------------------------


class TestSearchJsonFlag:
    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_json_flag_outputs_valid_json(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json emits a JSON array with content, score, source_path, heading_context."""
        from grounded_code_mcp.embeddings import EmbeddingResult
        from grounded_code_mcp.vectorstore import SearchResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="q", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = [
            SearchResult(
                chunk_id="id1",
                content="This is content.",
                score=0.92,
                metadata={"source_path": "docs/guide.md", "heading_context": ["Guide", "Setup"]},
            )
        ]
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test query", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "This is content."
        assert data[0]["score"] == 0.92
        assert data[0]["source_path"] == "docs/guide.md"
        assert data[0]["heading_context"] == ["Guide", "Setup"]

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    @patch("grounded_code_mcp.vectorstore.create_vector_store")
    def test_search_json_no_results_outputs_empty_list(
        self,
        mock_create_store: MagicMock,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json with no results emits an empty JSON array."""
        from grounded_code_mcp.embeddings import EmbeddingResult

        mock_settings = MagicMock()
        mock_settings.vectorstore.collection_prefix = "grounded_"
        mock_settings_cls.load.return_value = mock_settings

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="q", embedding=[0.1] * 10, model="test"
        )
        mock_from_settings.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_collections.return_value = ["grounded_python"]
        mock_store.search.return_value = []
        mock_create_store.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "obscure query", "--json"])

        assert result.exit_code == 0
        assert _json.loads(result.output) == []

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.embeddings.EmbeddingClient.from_settings")
    def test_search_json_ollama_error_outputs_error_json(
        self,
        mock_from_settings: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json when Ollama is unavailable emits an error JSON object."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_settings_cls.load.return_value = MagicMock()

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "http://localhost:11434", "Connection refused"
        )
        mock_from_settings.return_value = mock_embedder

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "query", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert isinstance(data, list)
        assert "error" in data[0]


# ---------------------------------------------------------------------------
# Tests for new `search-code` command
# ---------------------------------------------------------------------------


class TestSearchCodeCommand:
    def test_search_code_command_exists(self) -> None:
        """search-code is registered and has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search-code", "--help"])
        assert result.exit_code == 0
        assert "code" in result.output.lower()

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._search_code_examples_impl")
    def test_search_code_json_output(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json emits a JSON array of code results."""
        mock_impl.return_value = [
            {
                "code": "def foo(): pass",
                "language": "python",
                "source_path": "docs/api.md",
                "heading_context": ["Functions"],
                "score": 0.91,
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["search-code", "function definition", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert len(data) == 1
        assert data[0]["language"] == "python"
        assert data[0]["code"] == "def foo(): pass"

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._search_code_examples_impl")
    def test_search_code_passes_language_and_n_results(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--language and -n are forwarded to the impl function."""
        mock_impl.return_value = []
        runner = CliRunner()
        runner.invoke(cli, ["search-code", "test", "--language", "python", "-n", "3", "--json"])
        mock_impl.assert_called_once_with("test", "python", 3)

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._search_code_examples_impl")
    def test_search_code_human_output_shows_language_and_code(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Human (non-JSON) output includes language tag and code content."""
        mock_impl.return_value = [
            {
                "code": "def bar(): return 42",
                "language": "python",
                "source_path": "docs/api.md",
                "heading_context": ["Functions"],
                "score": 0.85,
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["search-code", "return value"])

        assert result.exit_code == 0
        assert "python" in result.output.lower()
        assert "bar" in result.output

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._search_code_examples_impl")
    def test_search_code_no_results_json_outputs_empty_list(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json with no code results emits an empty JSON array."""
        mock_impl.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ["search-code", "obscure", "--json"])

        assert result.exit_code == 0
        assert _json.loads(result.output) == []


# ---------------------------------------------------------------------------
# Tests for new `list-sources` command
# ---------------------------------------------------------------------------


class TestListSourcesCommand:
    def test_list_sources_command_exists(self) -> None:
        """list-sources is registered and has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-sources", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._list_sources_impl")
    def test_list_sources_json_output(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json emits a JSON array of source metadata dicts."""
        mock_impl.return_value = [
            {
                "path": "sources/python/fastapi.pdf.md",
                "collection": "grounded_python",
                "file_type": "md",
                "title": "FastAPI Docs",
                "chunk_count": 42,
                "ingested_at": "2026-01-01T00:00:00",
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-sources", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert len(data) == 1
        assert data[0]["title"] == "FastAPI Docs"
        assert data[0]["chunk_count"] == 42

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._list_sources_impl")
    def test_list_sources_passes_collection_filter(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--collection bare suffix is forwarded to the impl function."""
        mock_impl.return_value = []
        runner = CliRunner()
        runner.invoke(cli, ["list-sources", "--collection", "python", "--json"])
        mock_impl.assert_called_once_with("python")

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._list_sources_impl")
    def test_list_sources_human_output_shows_title_and_chunks(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Human output renders a table with title and chunk count."""
        mock_impl.return_value = [
            {
                "path": "sources/python/fastapi.pdf.md",
                "collection": "grounded_python",
                "file_type": "md",
                "title": "FastAPI Docs",
                "chunk_count": 42,
                "ingested_at": "2026-01-01T00:00:00",
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-sources"])

        assert result.exit_code == 0
        assert "FastAPI Docs" in result.output
        assert "42" in result.output


# ---------------------------------------------------------------------------
# Tests for new `source-info` command
# ---------------------------------------------------------------------------


class TestSourceInfoCommand:
    def test_source_info_command_exists(self) -> None:
        """source-info is registered and has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["source-info", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._get_source_info_impl")
    def test_source_info_json_output(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json emits a JSON object with full source metadata."""
        mock_impl.return_value = {
            "path": "sources/python/fastapi.pdf.md",
            "collection": "grounded_python",
            "file_type": "md",
            "title": "FastAPI Docs",
            "page_count": 10,
            "chunk_count": 42,
            "sha256": "abc123",
            "ingested_at": "2026-01-01T00:00:00",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["source-info", "sources/python/fastapi.pdf.md", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert data["title"] == "FastAPI Docs"
        assert data["chunk_count"] == 42

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._get_source_info_impl")
    def test_source_info_not_found_json_contains_error_key(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json for a missing source emits a JSON object with an 'error' key."""
        mock_impl.return_value = {"error": "Source not found"}
        runner = CliRunner()
        result = runner.invoke(cli, ["source-info", "nonexistent/path.md", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert "error" in data

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._get_source_info_impl")
    def test_source_info_human_output_shows_title_and_chunks(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Human output renders a table with title and chunk count."""
        mock_impl.return_value = {
            "path": "sources/python/fastapi.pdf.md",
            "collection": "grounded_python",
            "file_type": "md",
            "title": "FastAPI Docs",
            "page_count": 10,
            "chunk_count": 42,
            "sha256": "abc123",
            "ingested_at": "2026-01-01T00:00:00",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["source-info", "sources/python/fastapi.pdf.md"])

        assert result.exit_code == 0
        assert "FastAPI Docs" in result.output
        assert "42" in result.output


# ---------------------------------------------------------------------------
# Tests for new `query-graph` command
# ---------------------------------------------------------------------------


class TestQueryGraphCommand:
    def test_query_graph_command_exists(self) -> None:
        """query-graph is registered and has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query-graph", "--help"])
        assert result.exit_code == 0
        assert "graph" in result.output.lower()

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._query_graph_impl")
    def test_query_graph_json_output(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--json emits a JSON object with graph traversal results."""
        mock_impl.return_value = {
            "matched_concept_ids": ["dependency-injection"],
            "matched_nodes": [
                {
                    "id": "dependency-injection",
                    "type": "pattern",
                    "domain": "design",
                    "description": "DI pattern",
                    "source_slug": "patterns/di",
                }
            ],
            "relationships": [
                {
                    "from": "dependency-injection",
                    "rel": "implements",
                    "to": "inversion-of-control",
                }
            ],
            "linked_sources": ["patterns/di"],
            "summary": "DI is a design pattern.",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["query-graph", "dependency injection", "--json"])

        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert "matched_nodes" in data
        assert data["matched_concept_ids"] == ["dependency-injection"]

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._query_graph_impl")
    def test_query_graph_passes_depth_and_domain(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """--depth and --domain are forwarded to the impl function."""
        mock_impl.return_value = {
            "matched_nodes": [],
            "relationships": [],
            "linked_sources": [],
            "summary": "",
        }
        runner = CliRunner()
        runner.invoke(
            cli, ["query-graph", "concept", "--depth", "3", "--domain", "design", "--json"]
        )
        mock_impl.assert_called_once_with("concept", 3, "design")

    @patch("grounded_code_mcp.__main__.Settings")
    @patch("grounded_code_mcp.server.initialize")
    @patch("grounded_code_mcp.server._query_graph_impl")
    def test_query_graph_human_output_shows_summary(
        self,
        mock_impl: MagicMock,
        mock_init: MagicMock,
        mock_settings_cls: MagicMock,
    ) -> None:
        """Human output includes the summary line from the impl result."""
        mock_impl.return_value = {
            "matched_concept_ids": [],
            "matched_nodes": [],
            "relationships": [],
            "linked_sources": [],
            "summary": "No concept graph is available.",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["query-graph", "unknown concept"])

        assert result.exit_code == 0
        assert "No concept graph" in result.output
