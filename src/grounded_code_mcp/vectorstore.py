"""Vector store abstraction with Qdrant and ChromaDB implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grounded_code_mcp.chunking import Chunk
    from grounded_code_mcp.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a vector search."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_path(self) -> str:
        """Get the source path from metadata."""
        return str(self.metadata.get("source_path", ""))

    @property
    def heading_context(self) -> list[str]:
        """Get the heading context from metadata."""
        context = self.metadata.get("heading_context", [])
        if isinstance(context, list):
            return context
        return []


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def create_collection(
        self,
        name: str,
        *,
        embedding_dim: int = 1024,
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name.
            embedding_dim: Dimension of embedding vectors.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name.
        """

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if collection exists.
        """

    @abstractmethod
    def add_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks to a collection.

        Args:
            collection: Collection name.
            chunks: List of chunks to add.
            embeddings: Corresponding embedding vectors.
        """

    @abstractmethod
    def delete_chunks(self, collection: str, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            collection: Collection name.
            chunk_ids: List of chunk IDs to delete.
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        n_results: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            collection: Collection name.
            query_embedding: Query embedding vector.
            n_results: Maximum number of results.
            min_score: Minimum similarity score.
            filter_metadata: Optional metadata filters.

        Returns:
            List of SearchResult objects.
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List all collections.

        Returns:
            List of collection names.
        """

    @abstractmethod
    def collection_count(self, name: str) -> int:
        """Get the number of items in a collection.

        Args:
            name: Collection name.

        Returns:
            Number of items in the collection.
        """


class QdrantStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialize Qdrant store.

        Args:
            path: Path for persistent storage. If None, uses in-memory storage.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._distance = Distance
        self._vector_params = VectorParams

        if path is not None:
            self._client = QdrantClient(path=str(path))
        else:
            self._client = QdrantClient(":memory:")

    def create_collection(
        self,
        name: str,
        *,
        embedding_dim: int = 1024,
    ) -> None:
        """Create a new collection."""
        if self.collection_exists(name):
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=self._vector_params(
                size=embedding_dim,
                distance=self._distance.COSINE,
            ),
        )

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if self.collection_exists(name):
            self._client.delete_collection(collection_name=name)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self._client.collection_exists(collection_name=name)

    def add_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks to a collection."""
        from qdrant_client.models import PointStruct

        if not chunks:
            return

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = {
                "content": chunk.content,
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "heading_context": chunk.heading_context,
                "is_code": chunk.is_code,
                "code_language": chunk.code_language,
                "is_table": chunk.is_table,
            }
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=collection, points=points)

    def delete_chunks(self, collection: str, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        if not chunk_ids:
            return

        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=chunk_ids),  # type: ignore[arg-type]
        )

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        n_results: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)  # type: ignore[arg-type]

        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=n_results,
            score_threshold=min_score,
            query_filter=query_filter,
        )

        search_results: list[SearchResult] = []
        for r in results.points:
            payload = r.payload if r.payload else {}
            search_results.append(
                SearchResult(
                    chunk_id=str(r.id),
                    content=str(payload.get("content", "")),
                    score=r.score if r.score else 0.0,
                    metadata=dict(payload),
                )
            )
        return search_results

    def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]

    def collection_count(self, name: str) -> int:
        """Get the number of items in a collection."""
        if not self.collection_exists(name):
            return 0
        info = self._client.get_collection(collection_name=name)
        return info.points_count or 0

    def close(self) -> None:
        """Close the Qdrant client and release storage locks."""
        self._client.close()


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation (fallback)."""

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialize ChromaDB store.

        Args:
            path: Path for persistent storage. If None, uses ephemeral storage.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        if path is not None:
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def create_collection(
        self,
        name: str,
        *,
        embedding_dim: int = 1024,
    ) -> None:
        """Create a new collection."""
        # ChromaDB embedding_dim is inferred from first add
        _ = embedding_dim
        self._client.get_or_create_collection(name=name)

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if self.collection_exists(name):
            self._client.delete_collection(name=name)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        collections = self._client.list_collections()
        return name in [c.name for c in collections]

    def add_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks to a collection."""
        if not chunks:
            return

        coll = self._client.get_or_create_collection(name=collection)

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "is_code": chunk.is_code,
                "code_language": chunk.code_language or "",
                "is_table": chunk.is_table,
                # ChromaDB doesn't support list in metadata, serialize as string
                "heading_context": "|".join(chunk.heading_context),
            }
            for chunk in chunks
        ]

        coll.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def delete_chunks(self, collection: str, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        if not chunk_ids:
            return

        if not self.collection_exists(collection):
            return

        coll = self._client.get_collection(name=collection)
        coll.delete(ids=chunk_ids)

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        n_results: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return []

        where = filter_metadata if filter_metadata else None

        results = coll.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        ids_list = results.get("ids") or [[]]
        docs_list = results.get("documents") or [[]]
        metadatas_list = results.get("metadatas") or [[]]
        distances_list = results.get("distances") or [[]]

        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        distances = distances_list[0] if distances_list else []

        for i, chunk_id in enumerate(ids):
            # ChromaDB returns distances (lower is better), convert to similarity
            distance = float(distances[i]) if i < len(distances) else 0.0
            score = 1.0 - distance  # Convert distance to similarity

            if score < min_score:
                continue

            raw_meta = metadatas[i] if i < len(metadatas) else {}
            metadata: dict[str, Any] = dict(raw_meta) if raw_meta else {}
            # Deserialize heading_context
            if "heading_context" in metadata:
                hc = metadata["heading_context"]
                if isinstance(hc, str):
                    metadata["heading_context"] = hc.split("|") if hc else []

            search_results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=str(docs[i]) if i < len(docs) else "",
                    score=score,
                    metadata=metadata,
                )
            )

        return search_results

    def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def collection_count(self, name: str) -> int:
        """Get the number of items in a collection."""
        try:
            coll = self._client.get_collection(name=name)
            return coll.count()
        except Exception:
            return 0


def create_vector_store(settings: Settings) -> VectorStore:
    """Create a vector store based on settings.

    Args:
        settings: Application settings.

    Returns:
        VectorStore instance.
    """
    provider = settings.vectorstore.provider.lower()
    data_path = settings.knowledge_base.data_dir

    if provider == "qdrant":
        store_path = data_path / "qdrant"
        store_path.mkdir(parents=True, exist_ok=True)
        return QdrantStore(path=store_path)
    elif provider == "chromadb" or provider == "chroma":
        store_path = data_path / "chromadb"
        store_path.mkdir(parents=True, exist_ok=True)
        return ChromaStore(path=store_path)
    else:
        logger.warning("Unknown provider '%s', defaulting to Qdrant", provider)
        store_path = data_path / "qdrant"
        store_path.mkdir(parents=True, exist_ok=True)
        return QdrantStore(path=store_path)
