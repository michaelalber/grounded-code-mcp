"""Embeddings generation using Ollama."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ollama
from ollama import ResponseError

if TYPE_CHECKING:
    from grounded_code_mcp.config import OllamaSettings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Error connecting to Ollama server."""

    def __init__(self, host: str, message: str) -> None:
        self.host = host
        super().__init__(f"Cannot connect to Ollama at {host}: {message}")


class ModelNotFoundError(Exception):
    """Requested model is not available."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' not found. Run `ollama pull {model}` to download it.")


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    text: str
    embedding: list[float]
    model: str

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return len(self.embedding)


class EmbeddingClient:
    """Client for generating embeddings using Ollama."""

    def __init__(
        self,
        model: str = "snowflake-arctic-embed2",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize the embedding client.

        Args:
            model: Name of the embedding model to use.
            host: Ollama server URL.
        """
        self.model = model
        self.host = host
        self._client: ollama.Client | None = None

    @classmethod
    def from_settings(cls, settings: OllamaSettings) -> EmbeddingClient:
        """Create client from settings.

        Args:
            settings: Ollama settings.

        Returns:
            Configured EmbeddingClient.
        """
        return cls(model=settings.model, host=settings.host)

    @property
    def client(self) -> ollama.Client:
        """Get or create the Ollama client."""
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def health_check(self) -> dict[str, bool | str | int | None]:
        """Check if Ollama is running and model is available.

        Returns:
            Dict with 'healthy' bool and status details.
        """
        result: dict[str, bool | str | int | None] = {
            "healthy": False,
            "server_reachable": False,
            "model_available": False,
            "model": self.model,
            "host": self.host,
            "error": None,
        }

        try:
            # Check if server is reachable
            models_response = self.client.list()
            result["server_reachable"] = True

            # Check if our model is available
            # Support both old dict API and new typed API from ollama client
            models_list = getattr(models_response, "models", None)
            if models_list is None:
                models_list = models_response.get("models", [])  # type: ignore[union-attr]

            model_names: list[str] = []
            for m in models_list:
                name = getattr(m, "model", None) or (m.get("name", "") if isinstance(m, dict) else "")
                model_names.append(name)

            # Match against "model" or "model:latest"
            if any(
                n == self.model or n == f"{self.model}:latest" or n.split(":")[0] == self.model
                for n in model_names
            ):
                result["model_available"] = True
                result["healthy"] = True
            else:
                result["error"] = f"Model '{self.model}' not found"

        except Exception as e:
            result["error"] = str(e)

        return result

    def ensure_ready(self) -> None:
        """Ensure Ollama is ready for embedding generation.

        Raises:
            OllamaConnectionError: If server is not reachable.
            ModelNotFoundError: If model is not available.
        """
        health = self.health_check()

        if not health["server_reachable"]:
            raise OllamaConnectionError(
                self.host,
                str(health.get("error", "Server not responding")),
            )

        if not health["model_available"]:
            raise ModelNotFoundError(self.model)

    def embed(self, text: str, *, is_query: bool = False) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.
            is_query: If True, prepend query prefix for retrieval models
                (e.g. snowflake-arctic-embed2 uses "query: " prefix).

        Returns:
            EmbeddingResult with the embedding vector.

        Raises:
            OllamaConnectionError: If server is not reachable.
            ModelNotFoundError: If model is not available.
        """
        input_text = f"query: {text}" if is_query else text
        try:
            response = self.client.embed(model=self.model, input=input_text)
            embeddings = response.get("embeddings", [[]])
            embedding = embeddings[0] if embeddings else []

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
            )
        except ResponseError as e:
            if "not found" in str(e).lower():
                raise ModelNotFoundError(self.model) from e
            raise
        except Exception as e:
            if "connection" in str(e).lower():
                raise OllamaConnectionError(self.host, str(e)) from e
            raise

    def embed_many(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to embed per batch.
            show_progress: Whether to show progress output.
            is_query: If True, prepend query prefix for retrieval models.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            OllamaConnectionError: If server is not reachable.
            ModelNotFoundError: If model is not available.
        """
        if not texts:
            return []

        results: list[EmbeddingResult] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            if is_query:
                batch = [f"query: {t}" for t in batch]

            if show_progress:
                end = min(i + batch_size, total)
                logger.info("Embedding %d-%d of %d texts", i + 1, end, total)

            try:
                response = self.client.embed(model=self.model, input=batch)
                embeddings = response.get("embeddings", [])

                for text, embedding in zip(batch, embeddings, strict=False):
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            model=self.model,
                        )
                    )
            except ResponseError as e:
                if "not found" in str(e).lower():
                    raise ModelNotFoundError(self.model) from e
                raise
            except Exception as e:
                if "connection" in str(e).lower():
                    raise OllamaConnectionError(self.host, str(e)) from e
                raise

        return results

    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the current model.

        Returns:
            Number of dimensions in the embedding vector.

        Note:
            This generates a test embedding to determine dimensions.
        """
        result = self.embed("test")
        return result.dimensions


def get_helpful_error_message(error: Exception) -> str:
    """Get a helpful error message for common Ollama issues.

    Args:
        error: The exception that occurred.

    Returns:
        Human-readable error message with suggestions.
    """
    if isinstance(error, OllamaConnectionError):
        return (
            f"Cannot connect to Ollama at {error.host}.\n"
            "Please ensure Ollama is running:\n"
            "  - Linux: systemctl --user start ollama\n"
            "  - macOS: ollama serve\n"
            "  - Or check if Ollama is installed: https://ollama.ai"
        )

    if isinstance(error, ModelNotFoundError):
        return (
            f"Model '{error.model}' is not available.\nDownload it with: ollama pull {error.model}"
        )

    return str(error)
