"""Tests for the FastMCP server."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grounded_code_mcp.config import (
    KnowledgeBaseSettings,
    OllamaSettings,
    Settings,
    VectorStoreSettings,
)
from grounded_code_mcp.embeddings import EmbeddingResult
from grounded_code_mcp.manifest import Manifest, SourceEntry
from grounded_code_mcp.server import (
    _get_source_info_impl,
    _list_collections_impl,
    _list_sources_impl,
    _search_code_examples_impl,
    _search_knowledge_impl,
    initialize,
    run_server,
)
from grounded_code_mcp.vectorstore import SearchResult


class TestServerTools:
    """Tests for server MCP tools."""

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
            vectorstore=VectorStoreSettings(provider="qdrant"),
        )

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedding client."""
        embedder = MagicMock()
        embedder.ensure_ready.return_value = None
        embedder.embed.return_value = EmbeddingResult(
            text="test",
            embedding=[0.1] * 128,
            model="test",
        )
        return embedder

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        """Create a mock vector store."""
        store = MagicMock()
        store.list_collections.return_value = ["grounded_python", "grounded_docs"]
        store.collection_count.return_value = 10
        store.search.return_value = [
            SearchResult(
                chunk_id="chunk-1",
                content="Test content here",
                score=0.95,
                metadata={
                    "source_path": "docs/test.md",
                    "heading_context": ["Title", "Section"],
                    "is_code": False,
                },
            )
        ]
        return store

    @pytest.fixture
    def mock_manifest(self) -> Manifest:
        """Create a mock manifest."""
        manifest = Manifest()
        manifest.add_entry(
            SourceEntry(
                path="docs/test.md",
                sha256="abc123",
                collection="grounded_python",
                file_type="md",
                title="Test Document",
                chunk_count=5,
            )
        )
        return manifest

    def test_search_knowledge(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Test search_knowledge tool."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch(
                "grounded_code_mcp.server.create_vector_store",
                return_value=mock_store,
            ),
        ):
            results = _search_knowledge_impl("test query")

        # Mock returns 1 result per collection (2 collections), both within n_results=5
        assert len(results) == 2
        assert results[0]["content"] == "Test content here"
        assert results[0]["score"] == 0.95
        assert results[0]["source_path"] == "docs/test.md"

    def test_search_knowledge_with_collection(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Test search_knowledge with specific collection."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch(
                "grounded_code_mcp.server.create_vector_store",
                return_value=mock_store,
            ),
        ):
            results = _search_knowledge_impl("test query", collection="python")

        # Should search only the specified collection
        assert mock_store.search.call_count == 1
        call_args = mock_store.search.call_args
        assert call_args[0][0] == "grounded_python"
        # Should return formatted results from the single collection
        assert len(results) == 1
        assert results[0]["content"] == "Test content here"

    def test_search_knowledge_embedder_error(self, settings: Settings) -> None:
        """Test search_knowledge handles embedder errors."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "localhost", "Connection refused"
        )

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
        ):
            results = _search_knowledge_impl("test")

        assert len(results) == 1
        assert "error" in results[0]

    def test_search_code_examples(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Test search_code_examples tool."""
        mock_store.search.return_value = [
            SearchResult(
                chunk_id="code-1",
                content="def hello(): print('hi')",
                score=0.9,
                metadata={
                    "source_path": "examples/hello.py",
                    "heading_context": ["Examples"],
                    "is_code": True,
                    "code_language": "python",
                },
            )
        ]

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch(
                "grounded_code_mcp.server.create_vector_store",
                return_value=mock_store,
            ),
        ):
            results = _search_code_examples_impl("hello function")

        # Results from 2 collections, but limited to n_results
        assert len(results) >= 1
        assert "def hello()" in results[0]["code"]
        assert results[0]["language"] == "python"

    def test_search_code_examples_with_language_filter(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Test search_code_examples with language filter."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch(
                "grounded_code_mcp.server.create_vector_store",
                return_value=mock_store,
            ),
        ):
            _search_code_examples_impl("async request", language="python")

        # Check filter was applied
        mock_store.search.assert_called()
        call_args = mock_store.search.call_args
        assert call_args[1]["filter_metadata"]["code_language"] == "python"

    def test_list_collections(
        self,
        settings: Settings,
        mock_store: MagicMock,
        mock_manifest: Manifest,
    ) -> None:
        """Test list_collections tool."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
            patch(
                "grounded_code_mcp.server.create_vector_store",
                return_value=mock_store,
            ),
        ):
            results = _list_collections_impl()

        assert len(results) == 2
        assert results[0]["name"] == "grounded_python"
        assert results[0]["chunk_count"] == 10

    def test_list_sources(self, settings: Settings, mock_manifest: Manifest) -> None:
        """Test list_sources tool."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
        ):
            results = _list_sources_impl()

        assert len(results) == 1
        assert results[0]["path"] == "docs/test.md"
        assert results[0]["title"] == "Test Document"

    def test_list_sources_with_collection_filter(
        self, settings: Settings, mock_manifest: Manifest
    ) -> None:
        """Test list_sources with collection filter."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
        ):
            results = _list_sources_impl(collection="python")

        assert len(results) == 1

    def test_get_source_info(self, settings: Settings, mock_manifest: Manifest) -> None:
        """Test get_source_info tool."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
        ):
            result = _get_source_info_impl("docs/test.md")

        assert result["path"] == "docs/test.md"
        assert result["title"] == "Test Document"
        assert result["chunk_count"] == 5

    def test_get_source_info_not_found(self, settings: Settings, mock_manifest: Manifest) -> None:
        """Test get_source_info with nonexistent source."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
        ):
            result = _get_source_info_impl("nonexistent.md")

        assert "error" in result


class TestInitialize:
    """Tests for server initialization."""

    def test_initialize_with_settings(self, temp_dir: Path) -> None:
        """Test initialization with provided settings."""
        import grounded_code_mcp.server as server_module

        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=temp_dir / "sources",
                data_dir=temp_dir / "data",
            )
        )
        (temp_dir / "sources").mkdir()
        (temp_dir / "data").mkdir()

        try:
            initialize(settings)

            assert server_module._settings is not None
            assert server_module._settings.knowledge_base.sources_dir == temp_dir / "sources"
        finally:
            # Reset module globals to prevent state leaking into other tests
            server_module._settings = None
            server_module._embedder = None
            server_module._manifest = None


class TestRunServer:
    """Tests for run_server() transport parameters."""

    def test_run_server_default_uses_stdio(self) -> None:
        """run_server() with no transport calls mcp.run() with no transport args."""
        with (
            patch("grounded_code_mcp.server.initialize") as mock_init,
            patch("grounded_code_mcp.server.mcp") as mock_mcp,
        ):
            run_server()

        mock_init.assert_called_once()
        mock_mcp.run.assert_called_once_with()

    def test_run_server_streamable_http_forwards_kwargs(self) -> None:
        """run_server() forwards transport/host/port to mcp.run()."""
        with (
            patch("grounded_code_mcp.server.initialize"),
            patch("grounded_code_mcp.server.mcp") as mock_mcp,
        ):
            run_server(
                transport="streamable-http",
                host="0.0.0.0",  # noqa: S104 — testing explicit opt-in
                port=8080,
            )

        mock_mcp.run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104 — testing explicit opt-in
            port=8080,
        )

    def test_run_server_streamable_http_default_host(self) -> None:
        """run_server() defaults host to 127.0.0.1 for streamable-http."""
        with (
            patch("grounded_code_mcp.server.initialize"),
            patch("grounded_code_mcp.server.mcp") as mock_mcp,
        ):
            run_server(transport="streamable-http")

        mock_mcp.run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=8080,
        )
