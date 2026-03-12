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

    def test_max_file_size_mb_default(self) -> None:
        """Test max_file_size_mb defaults to 200."""
        settings = KnowledgeBaseSettings()
        assert settings.max_file_size_mb == 200

    def test_max_file_size_mb_configurable(self) -> None:
        """Test max_file_size_mb can be set."""
        settings = KnowledgeBaseSettings(max_file_size_mb=50)
        assert settings.max_file_size_mb == 50

    def test_pdf_page_batch_size_default_is_zero(self) -> None:
        """Test pdf_page_batch_size defaults to 0 (disabled)."""
        settings = KnowledgeBaseSettings()
        assert settings.pdf_page_batch_size == 0

    def test_pdf_page_batch_size_configurable(self) -> None:
        """Test pdf_page_batch_size can be set."""
        settings = KnowledgeBaseSettings(pdf_page_batch_size=50)
        assert settings.pdf_page_batch_size == 50


class TestOllamaSettings:
    """Tests for OllamaSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = OllamaSettings()
        assert settings.model == "snowflake-arctic-embed2"
        assert settings.host == "http://localhost:11434"
        assert settings.embedding_dim == 1024

    def test_context_length_default(self) -> None:
        """Test context_length defaults to 8192."""
        settings = OllamaSettings()
        assert settings.context_length == 8192


class TestChunkingSettings:
    """Tests for ChunkingSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = ChunkingSettings()
        assert settings.text_chunk_size == 1000
        assert settings.text_chunk_max_size == 1500
        assert settings.text_chunk_overlap == 200
        assert settings.max_code_chunk_size == 3000

    def test_ingest_batch_size_default(self) -> None:
        """Test ingest_batch_size defaults to 50."""
        settings = ChunkingSettings()
        assert settings.ingest_batch_size == 50

    def test_ingest_batch_size_configurable(self) -> None:
        """Test ingest_batch_size can be set."""
        settings = ChunkingSettings(ingest_batch_size=10)
        assert settings.ingest_batch_size == 10


class TestVectorStoreSettings:
    """Tests for VectorStoreSettings."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = VectorStoreSettings()
        assert settings.provider == "qdrant"
        assert settings.collection_prefix == "grounded_"
        assert settings.qdrant_url is None

    def test_qdrant_url_configurable(self) -> None:
        """Test that qdrant_url can be set for Docker/remote connections."""
        settings = VectorStoreSettings(qdrant_url="http://localhost:6333")
        assert settings.qdrant_url == "http://localhost:6333"

    def test_qdrant_url_defaults_to_none(self) -> None:
        settings = VectorStoreSettings()
        assert settings.qdrant_url is None

    def test_qdrant_url_from_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / "config.toml"
        toml.write_text('[vectorstore]\nqdrant_url = "http://localhost:6333"\n')
        settings = Settings.from_toml(toml)
        assert settings.vectorstore.qdrant_url == "http://localhost:6333"


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
        assert settings.ollama.model == "snowflake-arctic-embed2"
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
        no_user_cfg = temp_dir / "no-user.toml"

        settings = Settings.load(config_path, user_config_path=no_user_cfg)
        assert settings.ollama.model == "snowflake-arctic-embed2"

    def test_load_fallback_to_defaults(self, temp_dir: Path) -> None:
        """Test load() falls back to defaults when no config found."""
        no_user_cfg = temp_dir / "no-user.toml"
        settings = Settings.load(temp_dir / "nonexistent.toml", user_config_path=no_user_cfg)
        assert settings.ollama.model == "snowflake-arctic-embed2"

    def test_load_merges_user_config_over_project(
        self, temp_dir: Path, sample_config_toml: str
    ) -> None:
        """User config values override project config values."""
        project_cfg = temp_dir / "config.toml"
        project_cfg.write_text(sample_config_toml)

        user_cfg = temp_dir / "user.toml"
        user_cfg.write_text('[ollama]\nhost = "http://remote-host:11434"\n')

        settings = Settings.load(project_cfg, user_config_path=user_cfg)
        assert settings.ollama.host == "http://remote-host:11434"
        assert settings.ollama.model == "snowflake-arctic-embed2"  # project default kept

    def test_load_merges_collections_union(self, temp_dir: Path, sample_config_toml: str) -> None:
        """Collections from project and user config are merged (union), not replaced."""
        project_cfg = temp_dir / "config.toml"
        project_cfg.write_text(
            sample_config_toml + '\n[collections]\n"sources/internal" = "internal"\n'
        )

        user_cfg = temp_dir / "user.toml"
        user_cfg.write_text('[collections]\n"sources/custom" = "my_docs"\n')

        settings = Settings.load(project_cfg, user_config_path=user_cfg)
        assert settings.collections["sources/internal"] == "internal"
        assert settings.collections["sources/custom"] == "my_docs"

    def test_load_user_config_without_project_config(self, temp_dir: Path) -> None:
        """User config alone works when no project config is found."""
        user_cfg = temp_dir / "user.toml"
        user_cfg.write_text('[vectorstore]\nqdrant_url = "http://localhost:6333"\n')

        settings = Settings.load(temp_dir / "nonexistent.toml", user_config_path=user_cfg)
        assert settings.vectorstore.qdrant_url == "http://localhost:6333"

    def test_deep_merge_scalar_override(self) -> None:
        """Scalar values in override replace base values."""
        result = Settings._deep_merge({"a": 1, "b": 2}, {"b": 99, "c": 3})
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_deep_merge_nested_dicts_merged(self) -> None:
        """Nested dicts are merged recursively, not replaced wholesale."""
        base = {"ollama": {"host": "http://localhost", "model": "m1"}}
        override = {"ollama": {"host": "http://remote"}}
        result = Settings._deep_merge(base, override)
        assert result["ollama"]["host"] == "http://remote"
        assert result["ollama"]["model"] == "m1"  # preserved from base

    def test_deep_merge_collections_combined(self) -> None:
        """[collections] dicts from base and override are combined."""
        base = {"collections": {"sources/a": "col_a"}}
        override = {"collections": {"sources/b": "col_b"}}
        result = Settings._deep_merge(base, override)
        assert result["collections"] == {"sources/a": "col_a", "sources/b": "col_b"}

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
