"""Configuration loading and validation for grounded-code-mcp."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class KnowledgeBaseSettings(BaseModel):
    """Settings for the knowledge base paths."""

    sources_dir: Path = Field(default=Path("sources"))
    data_dir: Path = Field(default=Path(".grounded-code-mcp"))
    manifest_file: str = Field(default="manifest.json")
    max_file_size_mb: int = Field(
        default=200,
        description="Maximum file size in MB. Files larger than this are skipped. 0 disables.",
    )
    pdf_page_batch_size: int = Field(
        default=0,
        description=(
            "Split large PDFs into batches of this many pages before passing to Docling. "
            "Keeps peak memory bounded for 200MB+ files. 0 disables batching."
        ),
    )

    @property
    def manifest_path(self) -> Path:
        """Full path to the manifest file."""
        return self.data_dir / self.manifest_file


class OllamaSettings(BaseModel):
    """Settings for Ollama embeddings."""

    model: str = Field(default="snowflake-arctic-embed2")
    host: str = Field(default="http://localhost:11434")
    embedding_dim: int = Field(default=1024)
    context_length: int = Field(default=8192)


class ChunkingSettings(BaseModel):
    """Settings for document chunking."""

    text_chunk_size: int = Field(default=1000)
    text_chunk_max_size: int = Field(default=1500)
    text_chunk_overlap: int = Field(default=200)
    max_code_chunk_size: int = Field(default=3000)
    ingest_batch_size: int = Field(
        default=50,
        description="Number of chunks to embed and store per batch during ingestion.",
    )


class VectorStoreSettings(BaseModel):
    """Settings for the vector store."""

    provider: str = Field(default="qdrant")
    collection_prefix: str = Field(default="grounded_")
    qdrant_url: str | None = Field(default=None)


class Settings(BaseModel):
    """Application settings loaded from config.toml."""

    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    collections: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path) -> Settings:
        """Load settings from a TOML file.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            Settings instance with values from the file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            tomllib.TOMLDecodeError: If the TOML is invalid.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge override into base.

        Scalar values from override replace base values. Dict values are merged
        recursively, so [collections] entries from both configs are combined
        rather than the user config replacing the project config wholesale.

        Args:
            base: Base dictionary (e.g. from project config.toml).
            override: Override dictionary (e.g. from user config.toml).

        Returns:
            New dict with override values merged into base.
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        user_config_path: Path | None = None,
    ) -> Settings:
        """Load settings by merging project config with user config.

        Load order (later values override earlier):
        1. Built-in defaults
        2. Project config: explicit ``config_path``, or ``./config.toml``
        3. User config: ``~/.config/grounded-code-mcp/config.toml``

        The ``[collections]`` table is merged (union) across both files so
        users can add private collections without duplicating the project list.

        Args:
            config_path: Explicit path to the project config file. Defaults to
                ``./config.toml`` in the current working directory.
            user_config_path: Path to the user config file. Defaults to
                ``~/.config/grounded-code-mcp/config.toml``. Pass a
                non-existent path to disable user config (useful in tests).

        Returns:
            Settings instance with merged configuration.
        """
        data: dict[str, Any] = {}

        # Layer 1: project config
        project_path = config_path if config_path is not None else Path.cwd() / "config.toml"
        if project_path.exists():
            with open(project_path, "rb") as f:
                data = tomllib.load(f)

        # Layer 2: user config — always overlaid on top of project config
        resolved_user_path = (
            user_config_path
            if user_config_path is not None
            else Path.home() / ".config" / "grounded-code-mcp" / "config.toml"
        )
        if resolved_user_path.exists():
            with open(resolved_user_path, "rb") as f:
                user_data = tomllib.load(f)
            data = cls._deep_merge(data, user_data)

        return cls.model_validate(data) if data else cls()

    def get_collection_name(self, source_path: Path) -> str:
        """Get the collection name for a source path.

        Args:
            source_path: Path to a source file or directory.

        Returns:
            Collection name based on mapping or derived from path.
        """
        source_str = str(source_path)

        for prefix, collection in self.collections.items():
            if source_str == prefix or source_str.startswith(prefix + "/"):
                return f"{self.vectorstore.collection_prefix}{collection}"

        # Fallback: use first subdirectory under sources_dir
        try:
            relative = source_path.relative_to(self.knowledge_base.sources_dir)
            parts = relative.parts
            if parts:
                return f"{self.vectorstore.collection_prefix}{parts[0]}"
        except ValueError:
            pass

        return f"{self.vectorstore.collection_prefix}default"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.knowledge_base.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_base.sources_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary for display."""
        return self.model_dump()
