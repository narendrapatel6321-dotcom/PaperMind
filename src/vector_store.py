"""FAISS-backed vector store for dense retrieval.

Stores normalized document embeddings in a FAISS inner-product index.
Because embeddings are L2-normalized, inner product is equivalent to
cosine similarity.

The corresponding DocumentChunk objects are stored alongside the FAISS
index so that retrieval preserves document provenance and metadata.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_FILE = "faiss.index"
_CHUNKS_FILE = "chunks.pkl"


class FAISSVectorStore:
    """Dense vector store backed by FAISS IndexFlatIP."""

    def __init__(self) -> None:
        """Initialize an empty vector store."""
        self._index = None
        self._chunks: list = []

    @property
    def is_built(self) -> bool:
        """Return True if an index has been built or loaded."""
        return self._index is not None

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self._index is None:
            return 0

        return self._index.ntotal

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list,
    ) -> None:
        """Build the FAISS index from document embeddings.

        Args:
            embeddings: Array of shape (N, embedding_dim).
                        Embeddings should be L2-normalized.
            chunks: DocumentChunk objects corresponding to embeddings.

        Raises:
            ValueError: If inputs are invalid or lengths do not match.
            ImportError: If faiss-cpu is not installed.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D, got shape {embeddings.shape}"
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunks "
                f"({len(chunks)}) must have equal length."
            )

        if len(embeddings) == 0:
            raise ValueError("Cannot build an index from empty embeddings.")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        embeddings = np.asarray(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]

        # Inner product is cosine similarity when vectors are normalized.
        self._index = faiss.IndexFlatIP(dimension)

        self._index.add(embeddings)

        # Keep the exact same ordering as the FAISS vectors.
        self._chunks = list(chunks)

        logger.info(
            "Built FAISS index: %d vectors, dimension=%d",
            self._index.ntotal,
            dimension,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> list[tuple]:
        """Search for the top-k most similar document chunks.

        Args:
            query_embedding: Query vector of shape (dim,) or (1, dim).
            k: Number of results to return.

        Returns:
            List of:
                (DocumentChunk, similarity_score)

            Results are ordered from highest to lowest similarity.

        Raises:
            RuntimeError: If the index has not been built or loaded.
            ValueError: If k is invalid or dimensions do not match.
        """
        if not self.is_built:
            raise RuntimeError(
                "Index is empty. Call build() or load() first."
            )

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError(
                "query_embedding must have shape (dim,) or (1, dim)."
            )

        if query_embedding.shape[1] != self._index.d:
            raise ValueError(
                f"Query dimension ({query_embedding.shape[1]}) does not "
                f"match index dimension ({self._index.d})."
            )

        # FAISS cannot return more meaningful results than the number
        # of vectors actually stored.
        k = min(k, self.size)

        scores, indices = self._index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if 0 <= index < len(self._chunks):
                chunk = self._chunks[int(index)]
                results.append(
                    (chunk, float(score))
                )

        return results

    def save(self, directory: Path) -> None:
        """Save the FAISS index and chunk metadata to disk.

        Creates:

            directory/
                faiss.index
                chunks.pkl
        """
        if not self.is_built:
            raise RuntimeError(
                "Nothing to save. Build the index first."
            )

        import faiss

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = directory / _INDEX_FILE
        chunks_path = directory / _CHUNKS_FILE

        faiss.write_index(
            self._index,
            str(index_path),
        )

        with open(chunks_path, "wb") as file:
            pickle.dump(
                self._chunks,
                file,
            )

        logger.info(
            "Saved FAISS index with %d vectors to %s",
            self.size,
            directory,
        )

    def load(self, directory: Path) -> None:
        """Load a previously saved FAISS index and chunks.

        Args:
            directory: Directory containing:
                - faiss.index
                - chunks.pkl

        Raises:
            FileNotFoundError: If either file is missing.
        """
        directory = Path(directory)

        index_path = directory / _INDEX_FILE
        chunks_path = directory / _CHUNKS_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {chunks_path}"
            )

        import faiss

        self._index = faiss.read_index(
            str(index_path)
        )

        with open(chunks_path, "rb") as file:
            self._chunks = pickle.load(file)

        if self._index.ntotal != len(self._chunks):
            raise ValueError(
                "Loaded FAISS index and chunk metadata are inconsistent: "
                f"{self._index.ntotal} vectors vs "
                f"{len(self._chunks)} chunks."
            )

        logger.info(
            "Loaded FAISS index: %d vectors, dimension=%d",
            self._index.ntotal,
            self._index.d,
        )
