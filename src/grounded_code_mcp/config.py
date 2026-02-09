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

    @property
    def manifest_path(self) -> Path:
        """Full path to the manifest file."""
        return self.data_dir / self.manifest_file


class OllamaSettings(BaseModel):
    """Settings for Ollama embeddings."""

    model: str = Field(default="mxbai-embed-large")
    host: str = Field(default="http://localhost:11434")
    embedding_dim: int = Field(default=1024)


class ChunkingSettings(BaseModel):
    """Settings for document chunking."""

    text_chunk_size: int = Field(default=1000)
    text_chunk_max_size: int = Field(default=1500)
    text_chunk_overlap: int = Field(default=200)
    max_code_chunk_size: int = Field(default=3000)


class VectorStoreSettings(BaseModel):
    """Settings for the vector store."""

    provider: str = Field(default="qdrant")
    collection_prefix: str = Field(default="grounded_")


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

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """Load settings from config file or use defaults.

        Searches for config in:
        1. Explicit path if provided
        2. ./config.toml
        3. ~/.config/grounded-code-mcp/config.toml

        Args:
            config_path: Optional explicit path to config file.

        Returns:
            Settings instance.
        """
        search_paths: list[Path] = []

        if config_path:
            search_paths.append(config_path)

        search_paths.extend(
            [
                Path.cwd() / "config.toml",
                Path.home() / ".config" / "grounded-code-mcp" / "config.toml",
            ]
        )

        for path in search_paths:
            if path.exists():
                return cls.from_toml(path)

        return cls()

    def get_collection_name(self, source_path: Path) -> str:
        """Get the collection name for a source path.

        Args:
            source_path: Path to a source file or directory.

        Returns:
            Collection name based on mapping or derived from path.
        """
        source_str = str(source_path)

        for prefix, collection in self.collections.items():
            if source_str.startswith(prefix):
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
