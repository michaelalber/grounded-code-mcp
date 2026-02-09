"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from grounded_code_mcp.config import (
    ChunkingSettings,
    KnowledgeBaseSettings,
    OllamaSettings,
    Settings,
    VectorStoreSettings,
)


class TestKnowledgeBaseSettings:
    """Tests for KnowledgeBaseSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = KnowledgeBaseSettings()
        assert settings.sources_dir == Path("sources")
        assert settings.data_dir == Path(".grounded-code-mcp")
        assert settings.manifest_file == "manifest.json"

    def test_manifest_path(self) -> None:
        """Test manifest_path property."""
        settings = KnowledgeBaseSettings()
        assert settings.manifest_path == Path(".grounded-code-mcp/manifest.json")


class TestOllamaSettings:
    """Tests for OllamaSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = OllamaSettings()
        assert settings.model == "mxbai-embed-large"
        assert settings.host == "http://localhost:11434"
        assert settings.embedding_dim == 1024


class TestChunkingSettings:
    """Tests for ChunkingSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = ChunkingSettings()
        assert settings.text_chunk_size == 1000
        assert settings.text_chunk_max_size == 1500
        assert settings.text_chunk_overlap == 200
        assert settings.max_code_chunk_size == 3000


class TestVectorStoreSettings:
    """Tests for VectorStoreSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = VectorStoreSettings()
        assert settings.provider == "qdrant"
        assert settings.collection_prefix == "grounded_"


class TestSettings:
    """Tests for Settings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = Settings()
        assert isinstance(settings.knowledge_base, KnowledgeBaseSettings)
        assert isinstance(settings.ollama, OllamaSettings)
        assert isinstance(settings.chunking, ChunkingSettings)
        assert isinstance(settings.vectorstore, VectorStoreSettings)
        assert settings.collections == {}

    def test_from_toml(self, temp_dir: Path, sample_config_toml: str) -> None:
        """Test loading from TOML file."""
        config_path = temp_dir / "config.toml"
        config_path.write_text(sample_config_toml)

        settings = Settings.from_toml(config_path)

        assert settings.knowledge_base.sources_dir == Path("sources")
        assert settings.knowledge_base.data_dir == Path(".grounded-code-mcp")
        assert settings.ollama.model == "mxbai-embed-large"
        assert settings.chunking.text_chunk_size == 1000
        assert settings.vectorstore.provider == "qdrant"

    def test_from_toml_missing_file(self, temp_dir: Path) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Settings.from_toml(temp_dir / "nonexistent.toml")

    def test_load_with_explicit_path(self, temp_dir: Path, sample_config_toml: str) -> None:
        """Test load() with explicit path."""
        config_path = temp_dir / "myconfig.toml"
        config_path.write_text(sample_config_toml)

        settings = Settings.load(config_path)
        assert settings.ollama.model == "mxbai-embed-large"

    def test_load_fallback_to_defaults(self, temp_dir: Path) -> None:
        """Test load() falls back to defaults when no config found."""
        # Use a non-existent path to force defaults
        settings = Settings.load(temp_dir / "nonexistent.toml")
        assert settings.ollama.model == "mxbai-embed-large"

    def test_get_collection_name_from_mapping(self) -> None:
        """Test collection name from explicit mapping."""
        settings = Settings(
            collections={"sources/python": "python"},
            vectorstore=VectorStoreSettings(collection_prefix="test_"),
        )
        result = settings.get_collection_name(Path("sources/python/docs/file.md"))
        assert result == "test_python"

    def test_get_collection_name_fallback(self) -> None:
        """Test collection name fallback to directory name."""
        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(sources_dir=Path("sources")),
            vectorstore=VectorStoreSettings(collection_prefix="test_"),
        )
        result = settings.get_collection_name(Path("sources/newcollection/file.md"))
        assert result == "test_newcollection"

    def test_ensure_directories(self, temp_dir: Path) -> None:
        """Test directory creation."""
        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(
                sources_dir=temp_dir / "sources",
                data_dir=temp_dir / "data",
            )
        )
        settings.ensure_directories()

        assert (temp_dir / "sources").exists()
        assert (temp_dir / "data").exists()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        settings = Settings()
        result = settings.to_dict()

        assert "knowledge_base" in result
        assert "ollama" in result
        assert "chunking" in result
        assert "vectorstore" in result
