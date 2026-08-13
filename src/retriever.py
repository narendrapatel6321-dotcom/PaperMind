"""Query-time retrieval for the research-paper RAG system.

The retriever is responsible only for:
    1. Encoding a user query.
    2. Searching the vector store.
    3. Packaging retrieved chunks and similarity scores.

It does not perform reranking, generation, or evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.document_processor import DocumentChunk
from src.embedder import TextEmbedder
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Retrieved chunks together with their similarity scores.

    Attributes:
        chunks: Retrieved DocumentChunk objects ordered by similarity.
        scores: Cosine-similarity scores corresponding to ``chunks``.
    """

    chunks: list[DocumentChunk]
    scores: list[float]

    def __post_init__(self) -> None:
        """Validate that chunks and scores remain aligned."""
        if len(self.chunks) != len(self.scores):
            raise ValueError(
                "chunks and scores must have the same length."
            )

    def __len__(self) -> int:
        """Return the number of retrieved chunks."""
        return len(self.chunks)

    def format_context(self) -> str:
        """Format retrieved chunks for use in an LLM prompt.

        Source, section, page information, and similarity score are included
        so the generator has access to provenance information.
        """
        if not self.chunks:
            return ""

        parts: list[str] = []

        for rank, (chunk, score) in enumerate(
            zip(self.chunks, self.scores),
            start=1,
        ):
            metadata = chunk.metadata

            source = metadata.get(
                "title",
                chunk.source,
            )

            section = metadata.get(
                "section",
                "Unknown",
            )

            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")

            if page_start and page_end and page_start != page_end:
                pages = f"{page_start}-{page_end}"
            elif page_start:
                pages = page_start
            else:
                pages = "Unknown"

            parts.append(
                f"[{rank}] "
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Pages: {pages}\n"
                f"Similarity: {score:.4f}\n\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)


class DocumentRetriever:
    """Retrieve semantically relevant document chunks.

    The retriever combines a TextEmbedder and FAISSVectorStore.

    Args:
        embedder: TextEmbedder used to encode user queries.
        vector_store: Populated FAISSVectorStore.
        top_k: Maximum number of chunks returned per query.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: FAISSVectorStore,
        top_k: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievedContext:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language user question.

        Returns:
            RetrievedContext containing the retrieved chunks and scores.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If the vector store is not ready.
        """
        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        if not self.vector_store.is_built:
            raise RuntimeError(
                "Vector store is not ready. "
                "Build or load the vector store before retrieval."
            )

        # embed_query() is intentionally separate from document embedding.
        # This allows retrieval-specific query handling such as BGE prefixes.
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            k=self.top_k,
        )

        chunks = [chunk for chunk, _ in results]
        scores = [score for _, score in results]

        logger.info(
            "Retrieved %d chunk(s) for query '%s...' scores=%s",
            len(chunks),
            query[:60],
            [f"{score:.4f}" for score in scores],
        )

        return RetrievedContext(
            chunks=chunks,
            scores=scores,
        )
