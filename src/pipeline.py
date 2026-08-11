"""End-to-end RAG pipeline.

The pipeline orchestrates the individual components without implementing
their internal logic.

Indexing:
    DocumentProcessor
        -> TextEmbedder
        -> FAISSVectorStore
        -> persisted index

Querying:
    DocumentRetriever
        -> PromptBuilder
        -> RAGGenerator
        -> grounded answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import RAGConfig
from document_processor import DocumentProcessor, DocumentChunk
from embedder import TextEmbedder
from generator import PromptBuilder, RAGGenerator
from retriever import DocumentRetriever, RetrievedContext
from vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result returned by a single RAG query.

    Attributes:
        question: Original user question.
        answer: Generated answer.
        context: Retrieved chunks and similarity scores.
    """

    question: str
    answer: str
    context: RetrievedContext


class RAGPipeline:
    """Orchestrates document indexing and question answering."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize all RAG components.

        Args:
            config: Central project configuration.
        """
        self.config = config

        # --------------------------------------------------------------
        # Indexing components
        # --------------------------------------------------------------

        self.document_processor = DocumentProcessor()

        self.embedder = TextEmbedder(
            model_name=config.embedding_model,
            batch_size=config.embedding_batch_size,
            device=config.embedding_device,
            )

        self.vector_store = FAISSVectorStore()

        # --------------------------------------------------------------
        # Query-time components
        # --------------------------------------------------------------

        self.retriever = DocumentRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.top_k,
        )

        self.prompt_builder = PromptBuilder()

        self.generator = RAGGenerator(
            model_name=config.llm_model,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self) -> int:
        """Process, embed, and index the entire document corpus.

        The resulting FAISS index and chunk metadata are persisted to
        ``config.index_dir``.

        Returns:
            Number of document chunks indexed.
        """
        logger.info(
            "Starting document indexing from %s",
            self.config.data_dir,
        )

        # 1. Process documents into structure-aware chunks.
        chunks: list[DocumentChunk] = (
            self.document_processor.process_directory(
                self.config.data_dir
            )
        )

        if not chunks:
            raise ValueError(
                "DocumentProcessor returned no chunks. "
                "Check the input corpus and processing configuration."
            )

        logger.info(
            "Document processing produced %d chunks.",
            len(chunks),
        )

        # 2. Embed chunk text.
        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed_documents(texts)

        logger.info(
            "Generated embeddings with shape %s.",
            embeddings.shape,
        )

        # 3. Build FAISS index.
        self.vector_store.build(
            embeddings=embeddings,
            chunks=chunks,
        )

        # 4. Persist index + chunk metadata.
        self.vector_store.save(
            self.config.index_dir
        )

        logger.info(
            "Indexing complete: %d chunks saved to %s.",
            len(chunks),
            self.config.index_dir,
        )

        return len(chunks)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Load a previously persisted FAISS index.

        This skips document processing and document embedding.
        """
        logger.info(
            "Loading vector index from %s",
            self.config.index_dir,
        )

        self.vector_store.load(
            self.config.index_dir
        )

        logger.info(
            "Loaded vector index containing %d chunks.",
            self.vector_store.size,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Answer a question using retrieved document context.

        Args:
            question: Natural-language user question.

        Returns:
            RAGResult containing the answer and retrieved context.
        """
        question = question.strip()

        if not question:
            raise ValueError(
                "Question must not be empty."
            )

        if not self.vector_store.is_built:
            raise RuntimeError(
                "Vector index is not loaded. "
                "Call index() or load_index() before query()."
            )

        logger.info(
            "Processing query: '%s'",
            question,
        )

        # 1. Retrieve relevant chunks.
        context = self.retriever.retrieve(
            question
        )

        logger.info(
            "Retrieved %d chunks.",
            len(context),
        )

        # 2. Build grounded prompt.
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        # 3. Generate answer.
        answer = self.generator.generate(
            prompt
        )

        return RAGResult(
            question=question,
            answer=answer,
            context=context,
        )
