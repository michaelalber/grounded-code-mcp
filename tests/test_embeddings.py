"""Tests for embeddings generation."""

from unittest.mock import MagicMock, patch

import pytest
from ollama import ResponseError

from grounded_code_mcp.embeddings import (
    EmbeddingClient,
    EmbeddingResult,
    ModelNotFoundError,
    OllamaConnectionError,
    get_helpful_error_message,
)


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_dimensions(self) -> None:
        """Test dimensions property."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3, 0.4],
            model="test-model",
        )
        assert result.dimensions == 4

    def test_empty_embedding(self) -> None:
        """Test empty embedding dimensions."""
        result = EmbeddingResult(text="test", embedding=[], model="test-model")
        assert result.dimensions == 0


class TestEmbeddingClient:
    """Tests for EmbeddingClient."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        client = EmbeddingClient()
        assert client.model == "snowflake-arctic-embed2"
        assert client.host == "http://localhost:11434"

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        client = EmbeddingClient(
            model="custom-model",
            host="http://custom:8080",
        )
        assert client.model == "custom-model"
        assert client.host == "http://custom:8080"

    def test_from_settings(self) -> None:
        """Test creating from settings."""
        from grounded_code_mcp.config import OllamaSettings

        settings = OllamaSettings(
            model="test-model",
            host="http://test:11434",
        )

        client = EmbeddingClient.from_settings(settings)
        assert client.model == "test-model"
        assert client.host == "http://test:11434"

    def test_health_check_healthy(self) -> None:
        """Test health check when server and model are available."""
        client = EmbeddingClient(model="snowflake-arctic-embed2")

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"name": "snowflake-arctic-embed2:latest"}]}

        with patch.object(client, "_client", mock_ollama):
            result = client.health_check()

        assert result["healthy"] is True
        assert result["server_reachable"] is True
        assert result["model_available"] is True

    def test_health_check_model_missing(self) -> None:
        """Test health check when model is missing."""
        client = EmbeddingClient(model="missing-model")

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"name": "other-model:latest"}]}

        with patch.object(client, "_client", mock_ollama):
            result = client.health_check()

        assert result["healthy"] is False
        assert result["server_reachable"] is True
        assert result["model_available"] is False
        assert "not found" in str(result["error"]).lower()

    def test_health_check_server_unreachable(self) -> None:
        """Test health check when server is unreachable."""
        client = EmbeddingClient()

        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = Exception("Connection refused")

        with patch.object(client, "_client", mock_ollama):
            result = client.health_check()

        assert result["healthy"] is False
        assert result["server_reachable"] is False
        assert "Connection refused" in str(result["error"])

    def test_ensure_ready_success(self) -> None:
        """Test ensure_ready when everything is ok."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"name": "test-model:latest"}]}

        with patch.object(client, "_client", mock_ollama):
            # Should not raise
            client.ensure_ready()

    def test_ensure_ready_connection_error(self) -> None:
        """Test ensure_ready raises on connection error."""
        client = EmbeddingClient()

        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = Exception("Connection refused")

        with (
            patch.object(client, "_client", mock_ollama),
            pytest.raises(OllamaConnectionError),
        ):
            client.ensure_ready()

    def test_ensure_ready_model_not_found(self) -> None:
        """Test ensure_ready raises when model not found."""
        client = EmbeddingClient(model="missing-model")

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": []}

        with (
            patch.object(client, "_client", mock_ollama),
            pytest.raises(ModelNotFoundError),
        ):
            client.ensure_ready()

    def test_embed_success(self) -> None:
        """Test successful embedding generation for documents (no prefix)."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]}

        with patch.object(client, "_client", mock_ollama):
            result = client.embed("test text")

        assert result.text == "test text"
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.model == "test-model"
        mock_ollama.embed.assert_called_once_with(model="test-model", input="test text")

    def test_embed_query_prefix(self) -> None:
        """Test that is_query=True prepends 'query: ' prefix."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch.object(client, "_client", mock_ollama):
            result = client.embed("search terms", is_query=True)

        assert result.text == "search terms"
        mock_ollama.embed.assert_called_once_with(model="test-model", input="query: search terms")

    def test_embed_document_no_prefix(self) -> None:
        """Test that is_query=False (default) sends text without prefix."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch.object(client, "_client", mock_ollama):
            client.embed("document content", is_query=False)

        mock_ollama.embed.assert_called_once_with(model="test-model", input="document content")

    def test_embed_model_not_found(self) -> None:
        """Test embed raises when model not found."""
        client = EmbeddingClient(model="missing-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.side_effect = ResponseError("model not found")

        with (
            patch.object(client, "_client", mock_ollama),
            pytest.raises(ModelNotFoundError),
        ):
            client.embed("test")

    def test_embed_many_success(self) -> None:
        """Test batch embedding generation."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}

        texts = ["text1", "text2", "text3"]

        with patch.object(client, "_client", mock_ollama):
            results = client.embed_many(texts)

        assert len(results) == 3
        assert results[0].text == "text1"
        assert results[1].text == "text2"
        assert results[2].text == "text3"

    def test_embed_many_empty_list(self) -> None:
        """Test embed_many with empty list."""
        client = EmbeddingClient()
        results = client.embed_many([])
        assert results == []

    def test_embed_many_batching(self) -> None:
        """Test that embed_many respects batch size."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        # Return different embeddings for each batch call
        mock_ollama.embed.side_effect = [
            {"embeddings": [[0.1], [0.2]]},
            {"embeddings": [[0.3], [0.4]]},
            {"embeddings": [[0.5]]},
        ]

        texts = ["t1", "t2", "t3", "t4", "t5"]

        with patch.object(client, "_client", mock_ollama):
            results = client.embed_many(texts, batch_size=2)

        assert len(results) == 5
        assert mock_ollama.embed.call_count == 3

    def test_init_with_context_length(self) -> None:
        """Test constructor stores context_length."""
        client = EmbeddingClient(context_length=4096)
        assert client.context_length == 4096

    def test_max_chars_property(self) -> None:
        """Test max_chars = context_length * _CHARS_PER_TOKEN_ESTIMATE."""
        client = EmbeddingClient(context_length=8192)
        assert client.max_chars == 8192 * 3  # 24576

    def test_truncate_text_short(self) -> None:
        """Text under limit passes through unchanged."""
        client = EmbeddingClient(context_length=8192)
        short_text = "hello world"
        assert client._truncate_text(short_text) == short_text

    def test_truncate_text_long(self) -> None:
        """Text over limit is truncated to max_chars."""
        client = EmbeddingClient(context_length=100)
        max_chars = 100 * 3  # 300
        long_text = "x" * 500
        result = client._truncate_text(long_text)
        assert len(result) == max_chars

    def test_embed_truncates_long_text(self) -> None:
        """embed() truncates before sending to Ollama."""
        client = EmbeddingClient(model="test-model", context_length=100)
        max_chars = 100 * 3  # 300
        long_text = "a" * 500

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}

        with patch.object(client, "_client", mock_ollama):
            client.embed(long_text)

        # The text sent to Ollama should be truncated
        sent_text = mock_ollama.embed.call_args[1]["input"]
        assert len(sent_text) == max_chars

    def test_embed_many_truncates_long_texts(self) -> None:
        """embed_many() truncates each text before sending to Ollama."""
        client = EmbeddingClient(model="test-model", context_length=100)
        max_chars = 100 * 3  # 300
        texts = ["b" * 500, "short"]

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.1], [0.2]]}

        with patch.object(client, "_client", mock_ollama):
            client.embed_many(texts)

        sent_batch = mock_ollama.embed.call_args[1]["input"]
        assert len(sent_batch[0]) == max_chars
        assert sent_batch[1] == "short"

    def test_from_settings_passes_context_length(self) -> None:
        """from_settings() wires through context_length."""
        from grounded_code_mcp.config import OllamaSettings

        settings = OllamaSettings(context_length=4096)
        client = EmbeddingClient.from_settings(settings)
        assert client.context_length == 4096

    def test_get_embedding_dimensions(self) -> None:
        """Test getting embedding dimensions."""
        client = EmbeddingClient(model="test-model")

        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = {"embeddings": [[0.0] * 1024]}

        with patch.object(client, "_client", mock_ollama):
            dims = client.get_embedding_dimensions()

        assert dims == 1024


class TestGetHelpfulErrorMessage:
    """Tests for get_helpful_error_message function."""

    def test_connection_error_message(self) -> None:
        """Test message for connection error."""
        error = OllamaConnectionError("http://localhost:11434", "Connection refused")
        message = get_helpful_error_message(error)

        assert "Cannot connect" in message
        assert "localhost:11434" in message
        assert "systemctl" in message

    def test_model_not_found_message(self) -> None:
        """Test message for model not found."""
        error = ModelNotFoundError("snowflake-arctic-embed2")
        message = get_helpful_error_message(error)

        assert "not available" in message
        assert "ollama pull snowflake-arctic-embed2" in message

    def test_generic_error_message(self) -> None:
        """Test message for generic error."""
        error = RuntimeError("Something went wrong")
        message = get_helpful_error_message(error)

        assert message == "Something went wrong"


class TestExceptionClasses:
    """Tests for exception classes."""

    def test_ollama_connection_error(self) -> None:
        """Test OllamaConnectionError."""
        error = OllamaConnectionError("http://localhost:11434", "Connection refused")
        assert error.host == "http://localhost:11434"
        assert "localhost:11434" in str(error)
        assert "Connection refused" in str(error)

    def test_model_not_found_error(self) -> None:
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("my-model")
        assert error.model == "my-model"
        assert "my-model" in str(error)
        assert "ollama pull" in str(error)
