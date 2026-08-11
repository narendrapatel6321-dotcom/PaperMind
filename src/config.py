"""Configuration for the PaperMind RAG pipeline.

All fields can be overridden using environment variables with the RAG_
prefix.

Example:
    RAG_TOP_K=5 python main.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """Central configuration for the RAG pipeline.

    Chunking configuration intentionally lives inside DocumentProcessor,
    which is the single owner of document-ingestion behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    data_dir: Path = Path("data/papers")
    index_dir: Path = Path("index")
    output_dir: Path = Path("outputs")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str | None = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    top_k: int = 3

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def model_post_init(self, __context: object) -> None:
        """Validate configuration after Pydantic initialization."""
        if self.embedding_batch_size <= 0:
            raise ValueError(
                "embedding_batch_size must be greater than 0."
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

    def create_directories(self) -> None:
        """Create directories required by the pipeline."""
        for directory in (
            self.data_dir,
            self.index_dir,
            self.output_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
