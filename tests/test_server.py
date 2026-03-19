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
            collections={"sources/python": "python", "sources/docs": "docs"},
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


class TestInputValidation:
    """Tests for MCP tool input validation (M1-M4)."""

    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        return Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=sources_dir,
                data_dir=data_dir,
            ),
            ollama=OllamaSettings(model="test-model", embedding_dim=128),
            vectorstore=VectorStoreSettings(
                provider="qdrant", collection_prefix="grounded_"
            ),
            collections={"sources/python": "python", "sources/internal": "internal"},
        )

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.ensure_ready.return_value = None
        embedder.embed.return_value = EmbeddingResult(
            text="test", embedding=[0.1] * 128, model="test"
        )
        return embedder

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.list_collections.return_value = ["grounded_python", "grounded_internal"]
        store.search.return_value = []
        return store

    # M1: n_results clamping

    def test_search_knowledge_clamps_n_results_above_maximum(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """n_results > 50 must be clamped to 50 before querying the store."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl("query", n_results=1_000_000)

        for call in mock_store.search.call_args_list:
            assert call[1]["n_results"] <= 50

    def test_search_knowledge_clamps_n_results_below_minimum(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """n_results < 1 must be clamped to 1."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl("query", n_results=0)

        for call in mock_store.search.call_args_list:
            assert call[1]["n_results"] >= 1

    def test_search_knowledge_clamps_min_score_above_one(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """min_score > 1.0 must be clamped to 1.0."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl("query", min_score=99.0)

        for call in mock_store.search.call_args_list:
            assert call[1]["min_score"] <= 1.0

    def test_search_knowledge_clamps_min_score_below_zero(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """min_score < 0.0 must be clamped to 0.0."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl("query", min_score=-5.0)

        for call in mock_store.search.call_args_list:
            assert call[1]["min_score"] >= 0.0

    def test_search_code_examples_clamps_n_results_above_maximum(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """n_results > 50 must be clamped in search_code_examples too."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_code_examples_impl("query", n_results=9999)

        for call in mock_store.search.call_args_list:
            assert call[1]["n_results"] <= 50

    # M2: query length cap

    def test_search_knowledge_truncates_oversized_query(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """A query longer than MAX_QUERY_CHARS must be truncated before embedding."""
        oversized = "x" * 100_000

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl(oversized)

        embedded_text = mock_embedder.embed.call_args[0][0]
        assert len(embedded_text) <= 4_000

    def test_search_code_examples_truncates_oversized_query(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_code_examples must also truncate oversized queries."""
        oversized = "y" * 100_000

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_code_examples_impl(oversized)

        embedded_text = mock_embedder.embed.call_args[0][0]
        assert len(embedded_text) <= 4_000

    # M3: collection allowlist

    def test_search_knowledge_rejects_unknown_collection(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """An unrecognised collection name must return an error, not query the store."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            results = _search_knowledge_impl("query", collection="../../evil")

        assert len(results) == 1
        assert "error" in results[0]
        mock_store.search.assert_not_called()

    def test_search_knowledge_accepts_known_collection(
        self,
        settings: Settings,
        mock_embedder: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """A recognised collection suffix must pass through and query the store."""
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_knowledge_impl("query", collection="python")

        mock_store.search.assert_called_once()

    # M4: initialize idempotency (lock-protected re-init)

    def test_initialize_is_idempotent_when_already_initialized(
        self, temp_dir: Path
    ) -> None:
        """Calling initialize() a second time must not overwrite existing state."""
        import grounded_code_mcp.server as server_module

        first_settings = Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=temp_dir / "sources",
                data_dir=temp_dir / "data",
            )
        )
        (temp_dir / "sources").mkdir()
        (temp_dir / "data").mkdir()

        try:
            initialize(first_settings)
            captured = server_module._settings

            # Second call with different (None → defaults) settings must be a no-op
            initialize()

            assert server_module._settings is captured
        finally:
            server_module._settings = None
            server_module._embedder = None
            server_module._manifest = None


class TestLowSeverityFindings:
    """Regression tests for L1-L4 low-severity security findings."""

    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        return Settings(
            knowledge_base=KnowledgeBaseSettings(sources_dir=sources_dir, data_dir=data_dir),
            ollama=OllamaSettings(model="test-model", embedding_dim=128),
            vectorstore=VectorStoreSettings(provider="qdrant", collection_prefix="grounded_"),
            collections={"sources/python": "python"},
        )

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.list_collections.return_value = ["grounded_python"]
        store.search.return_value = []
        return store

    @pytest.fixture
    def mock_manifest(self) -> Manifest:
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

    # L1: Ollama host must not appear in tool error output

    def test_search_knowledge_ollama_error_does_not_expose_host(
        self, settings: Settings
    ) -> None:
        """OllamaConnectionError must not leak the host URL into the tool response."""
        from grounded_code_mcp.embeddings import OllamaConnectionError

        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.side_effect = OllamaConnectionError(
            "http://internal-host:11434", "Connection refused"
        )

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
        ):
            results = _search_knowledge_impl("query")

        assert len(results) == 1
        assert "error" in results[0]
        assert "internal-host" not in results[0]["error"]
        assert "11434" not in results[0]["error"]

    # L2: source_path must be capped and not echoed raw in error output

    def test_get_source_info_truncates_oversized_path(
        self, settings: Settings, mock_manifest: "Manifest"
    ) -> None:
        """A source_path longer than MAX_SOURCE_PATH_CHARS must not be echoed raw."""
        oversized = "x" * 10_000
        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._manifest", mock_manifest),
        ):
            result = _get_source_info_impl(oversized)

        assert "error" in result
        # The raw 10 000-char string must not appear verbatim in the error
        assert len(result["error"]) < 500

    # L3: unknown language must not be forwarded to the store

    def test_search_code_examples_ignores_unknown_language(
        self, settings: Settings, mock_store: MagicMock
    ) -> None:
        """An unrecognised language value must not be forwarded as a filter."""
        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="test", embedding=[0.1] * 128, model="test"
        )

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_code_examples_impl("query", language="'; DROP TABLE chunks; --")

        for call in mock_store.search.call_args_list:
            filter_meta = call[1].get("filter_metadata", {})
            assert "code_language" not in filter_meta

    def test_search_code_examples_passes_known_language_filter(
        self, settings: Settings, mock_store: MagicMock
    ) -> None:
        """A recognised language must still be forwarded as a filter."""
        mock_embedder = MagicMock()
        mock_embedder.ensure_ready.return_value = None
        mock_embedder.embed.return_value = EmbeddingResult(
            text="test", embedding=[0.1] * 128, model="test"
        )

        with (
            patch("grounded_code_mcp.server._settings", settings),
            patch("grounded_code_mcp.server._embedder", mock_embedder),
            patch("grounded_code_mcp.server.create_vector_store", return_value=mock_store),
        ):
            _search_code_examples_impl("query", language="python")

        call_args = mock_store.search.call_args
        assert call_args[1]["filter_metadata"]["code_language"] == "python"

    # L4: non-loopback HTTP bind must emit a warning

    def test_run_server_warns_when_http_transport_binds_to_non_loopback(self) -> None:
        """run_server() must log a warning when HTTP transport is bound to 0.0.0.0."""
        with (
            patch("grounded_code_mcp.server.initialize"),
            patch("grounded_code_mcp.server.mcp"),
            patch("grounded_code_mcp.server.logger") as mock_logger,
        ):
            run_server(transport="streamable-http", host="0.0.0.0")  # noqa: S104

        mock_logger.warning.assert_called()
        warn_msg = mock_logger.warning.call_args[0][0]
        non_loopback = "0.0.0.0"  # noqa: S104
        assert non_loopback in warn_msg or "loopback" in warn_msg.lower()

    def test_run_server_no_warning_for_loopback_host(self) -> None:
        """run_server() must not warn when bound to 127.0.0.1."""
        with (
            patch("grounded_code_mcp.server.initialize"),
            patch("grounded_code_mcp.server.mcp"),
            patch("grounded_code_mcp.server.logger") as mock_logger,
        ):
            run_server(transport="streamable-http", host="127.0.0.1")

        non_loopback = "0.0.0.0"  # noqa: S104
        for call in mock_logger.warning.call_args_list:
            assert "loopback" not in str(call).lower()
            assert non_loopback not in str(call)


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
