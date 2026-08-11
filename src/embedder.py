"""E5-small text embedding for the research-paper RAG system.

The embedder is responsible only for converting text into normalized
float32 vectors. Retrieval and vector storage are handled by separate
components.

E5-small-v2 expects:
    Documents: "passage: <text>"
    Queries:   "query: <text>"
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper around the E5-small-v2 SentenceTransformer model."""

    DEFAULT_MODEL = "intfloat/e5-small-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_name: SentenceTransformer model identifier.
            batch_size: Batch size used for document embedding.
            device: Device passed to SentenceTransformer. If None,
                SentenceTransformer selects the appropriate device.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self._model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the embedding model lazily."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self.model_name,
            self.device or "auto",
        )

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        logger.info(
            "Embedding model loaded. Dimension=%d",
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Document embedding
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed research-paper chunks.

        E5 expects document passages to use the ``passage:`` prefix.

        Args:
            texts: List of document/chunk texts.
            show_progress: Whether to display an encoding progress bar.

        Returns:
            Normalized float32 array with shape
            ``(len(texts), embedding_dim)``.

        Raises:
            TypeError: If texts is not a list of strings.
            ValueError: If any text is empty.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string.")

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not be empty.")

        if not texts:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )

        self._load()

        formatted_texts = [
            f"passage: {text.strip()}"
            for text in texts
        ]

        embeddings = self._model.encode(
            formatted_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query.

        E5 expects queries to use the ``query:`` prefix.

        Args:
            query: User's search/question text.

        Returns:
            Normalized float32 vector with shape
            ``(embedding_dim,)``.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        self._load()

        formatted_query = f"query: {query}"

        embedding = self._model.encode(
            formatted_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Backward-compatible alias for document embedding.

        New code should use ``embed_documents()`` instead.
        """
        if batch_size is None or batch_size == self.batch_size:
            return self.embed_documents(
                texts,
                show_progress=show_progress,
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string.")

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not be empty.")

        if not texts:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )

        self._load()

        formatted_texts = [
            f"passage: {text.strip()}"
            for text in texts
        ]

        embeddings = self._model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )
