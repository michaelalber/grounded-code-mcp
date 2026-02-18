"""Tests for vector store abstraction."""

import uuid
from pathlib import Path

import pytest

from grounded_code_mcp.chunking import Chunk
from grounded_code_mcp.vectorstore import (
    ChromaStore,
    QdrantStore,
    SearchResult,
    create_vector_store,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_source_path(self) -> None:
        """Test source_path property."""
        result = SearchResult(
            chunk_id="id",
            content="content",
            score=0.9,
            metadata={"source_path": "test.md"},
        )
        assert result.source_path == "test.md"

    def test_source_path_missing(self) -> None:
        """Test source_path when not in metadata."""
        result = SearchResult(chunk_id="id", content="content", score=0.9)
        assert result.source_path == ""

    def test_heading_context(self) -> None:
        """Test heading_context property."""
        result = SearchResult(
            chunk_id="id",
            content="content",
            score=0.9,
            metadata={"heading_context": ["Chapter 1", "Section A"]},
        )
        assert result.heading_context == ["Chapter 1", "Section A"]

    def test_heading_context_missing(self) -> None:
        """Test heading_context when not in metadata."""
        result = SearchResult(chunk_id="id", content="content", score=0.9)
        assert result.heading_context == []


class TestQdrantStore:
    """Tests for QdrantStore implementation."""

    @pytest.fixture
    def store(self) -> QdrantStore:
        """Create an in-memory Qdrant store."""
        return QdrantStore()  # In-memory

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        # Qdrant local client requires UUID-format IDs
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                content="First chunk content",
                chunk_index=0,
                source_path="test.md",
                heading_context=["Title", "Section 1"],
            ),
            Chunk(
                chunk_id=str(uuid.uuid4()),
                content="Second chunk content",
                chunk_index=1,
                source_path="test.md",
                heading_context=["Title", "Section 2"],
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Create sample embeddings (128-dim for speed)."""
        return [
            [0.1] * 128,
            [0.2] * 128,
        ]

    def test_create_collection(self, store: QdrantStore) -> None:
        """Test creating a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        assert store.collection_exists("test_collection")

    def test_create_collection_idempotent(self, store: QdrantStore) -> None:
        """Test that creating an existing collection doesn't raise."""
        store.create_collection("test_collection", embedding_dim=128)
        store.create_collection("test_collection", embedding_dim=128)
        assert store.collection_exists("test_collection")

    def test_delete_collection(self, store: QdrantStore) -> None:
        """Test deleting a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        store.delete_collection("test_collection")
        assert not store.collection_exists("test_collection")

    def test_delete_nonexistent_collection(self, store: QdrantStore) -> None:
        """Test deleting a nonexistent collection doesn't raise."""
        store.delete_collection("nonexistent")  # Should not raise

    def test_add_chunks(
        self,
        store: QdrantStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test adding chunks to a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        assert store.collection_count("test_collection") == 2

    def test_add_empty_chunks(self, store: QdrantStore) -> None:
        """Test adding empty chunk list."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", [], [])
        assert store.collection_count("test_collection") == 0

    def test_delete_chunks(
        self,
        store: QdrantStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test deleting chunks by ID."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        # Delete the first chunk by its ID
        store.delete_chunks("test_collection", [sample_chunks[0].chunk_id])
        assert store.collection_count("test_collection") == 1

    def test_search(
        self,
        store: QdrantStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test searching for similar chunks."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        # Search with query similar to first chunk
        query = [0.1] * 128
        results = store.search("test_collection", query, n_results=5)

        assert len(results) > 0
        # Check that the chunk ID matches one of the added chunks
        valid_ids = {c.chunk_id for c in sample_chunks}
        assert results[0].chunk_id in valid_ids
        assert results[0].score > 0

    def test_search_with_min_score(
        self,
        store: QdrantStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test search with minimum score filter."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        # All embeddings are parallel vectors ([0.1]*128, [0.2]*128),
        # so cosine similarity with [0.5]*128 is exactly 1.0
        query = [0.5] * 128
        results = store.search("test_collection", query, min_score=0.99)

        assert len(results) == 2, "Parallel vectors should have cosine similarity 1.0"
        assert all(r.score >= 0.99 for r in results)

    def test_list_collections(self, store: QdrantStore) -> None:
        """Test listing collections."""
        store.create_collection("collection_a", embedding_dim=128)
        store.create_collection("collection_b", embedding_dim=128)

        collections = store.list_collections()
        assert "collection_a" in collections
        assert "collection_b" in collections

    def test_collection_count(
        self,
        store: QdrantStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test getting collection count."""
        store.create_collection("test_collection", embedding_dim=128)
        assert store.collection_count("test_collection") == 0

        store.add_chunks("test_collection", sample_chunks, sample_embeddings)
        assert store.collection_count("test_collection") == 2

    def test_persistent_storage(
        self,
        temp_dir: Path,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test persistent storage."""
        store_path = temp_dir / "qdrant"
        store_path.mkdir()

        # Create and populate
        store1 = QdrantStore(path=store_path)
        store1.create_collection("persistent", embedding_dim=128)
        store1.add_chunks("persistent", sample_chunks, sample_embeddings)

        # Close the first store before opening a new one
        # Qdrant local client locks the storage directory
        store1.close()

        # Create new instance and verify data persists
        store2 = QdrantStore(path=store_path)
        assert store2.collection_exists("persistent")
        assert store2.collection_count("persistent") == 2


class TestChromaStore:
    """Tests for ChromaStore implementation."""

    @pytest.fixture
    def store(self) -> ChromaStore:
        """Create an ephemeral ChromaDB store."""
        return ChromaStore()  # Ephemeral

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        return [
            Chunk(
                chunk_id="chunk-1",
                content="First chunk content",
                chunk_index=0,
                source_path="test.md",
                heading_context=["Title", "Section 1"],
            ),
            Chunk(
                chunk_id="chunk-2",
                content="Second chunk content",
                chunk_index=1,
                source_path="test.md",
                heading_context=["Title", "Section 2"],
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Create sample embeddings (128-dim for speed)."""
        return [
            [0.1] * 128,
            [0.2] * 128,
        ]

    def test_create_collection(self, store: ChromaStore) -> None:
        """Test creating a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        assert store.collection_exists("test_collection")

    def test_delete_collection(self, store: ChromaStore) -> None:
        """Test deleting a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        store.delete_collection("test_collection")
        assert not store.collection_exists("test_collection")

    def test_add_chunks(
        self,
        store: ChromaStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test adding chunks to a collection."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        assert store.collection_count("test_collection") == 2

    def test_search(
        self,
        store: ChromaStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test searching for similar chunks."""
        store.create_collection("test_collection", embedding_dim=128)
        store.add_chunks("test_collection", sample_chunks, sample_embeddings)

        query = [0.1] * 128
        results = store.search("test_collection", query, n_results=5)

        assert len(results) > 0

    def test_list_collections(self, store: ChromaStore) -> None:
        """Test listing collections."""
        store.create_collection("collection_a", embedding_dim=128)
        store.create_collection("collection_b", embedding_dim=128)

        collections = store.list_collections()
        assert "collection_a" in collections
        assert "collection_b" in collections


class TestCreateVectorStore:
    """Tests for create_vector_store factory function."""

    def test_creates_qdrant_by_default(self, temp_dir: Path) -> None:
        """Test that Qdrant is created by default."""
        from grounded_code_mcp.config import (
            KnowledgeBaseSettings,
            Settings,
            VectorStoreSettings,
        )

        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(data_dir=temp_dir),
            vectorstore=VectorStoreSettings(provider="qdrant"),
        )

        store = create_vector_store(settings)
        assert isinstance(store, QdrantStore)

    def test_creates_chromadb(self, temp_dir: Path) -> None:
        """Test creating ChromaDB store."""
        from grounded_code_mcp.config import (
            KnowledgeBaseSettings,
            Settings,
            VectorStoreSettings,
        )

        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(data_dir=temp_dir),
            vectorstore=VectorStoreSettings(provider="chromadb"),
        )

        store = create_vector_store(settings)
        assert isinstance(store, ChromaStore)

    def test_unknown_provider_defaults_to_qdrant(self, temp_dir: Path) -> None:
        """Test that unknown provider defaults to Qdrant."""
        from grounded_code_mcp.config import (
            KnowledgeBaseSettings,
            Settings,
            VectorStoreSettings,
        )

        settings = Settings(
            knowledge_base=KnowledgeBaseSettings(data_dir=temp_dir),
            vectorstore=VectorStoreSettings(provider="unknown"),
        )

        store = create_vector_store(settings)
        assert isinstance(store, QdrantStore)
