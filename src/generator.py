"""Prompt construction and LLM-based answer generation for PaperMind.

PromptBuilder
    Converts retrieved document chunks into a grounded RAG prompt.

RAGGenerator
    Wraps a Hugging Face text-generation model with lazy loading.

The generator knows nothing about embeddings, FAISS, or document ingestion.
It receives a prompt and produces an answer.
"""

from __future__ import annotations

import logging

from retriever import RetrievedContext

logger = logging.getLogger(__name__)


_DEFAULT_FALLBACK = "Not found in the provided documents."

_PROMPT_TEMPLATE = """\
You are a research assistant answering questions about research papers.

Use ONLY the information contained in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or infer facts that are not supported by the context.
3. If the context does not contain enough information to answer the question,
   respond exactly with:
   "{fallback}"
4. Give a concise, direct answer.
5. When possible, mention the relevant source and section.

Context:
{context}

Question:
{question}

Answer:
"""


class PromptBuilder:
    """Build grounded prompts from a question and retrieved context."""

    def __init__(
        self,
        fallback_answer: str = _DEFAULT_FALLBACK,
    ) -> None:
        if not fallback_answer.strip():
            raise ValueError("fallback_answer must not be empty.")

        self.fallback_answer = fallback_answer.strip()

    def build(
        self,
        question: str,
        context: RetrievedContext,
    ) -> str:
        """Build the prompt sent to the language model.

        Args:
            question: User's natural-language question.
            context: Retrieved chunks and similarity scores.

        Returns:
            A complete grounded RAG prompt.

        Raises:
            ValueError: If the question is empty.
        """
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty.")

        formatted_context = context.format_context()

        if not formatted_context:
            formatted_context = (
                "No relevant documents were retrieved."
            )

        return _PROMPT_TEMPLATE.format(
            context=formatted_context,
            question=question,
            fallback=self.fallback_answer,
        )


class RAGGenerator:
    """Lazy-loading Hugging Face text-generation wrapper.

    The model is loaded only when ``generate()`` is first called.

    Args:
        model_name: Hugging Face model identifier or local model path.
        max_new_tokens: Maximum number of tokens generated for an answer.
        temperature: Sampling temperature. Use 0 for deterministic generation.
        device: Optional Hugging Face device specification.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        self._pipeline = None

    def _load(self) -> None:
        """Load the Hugging Face generation pipeline lazily."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Install it with: pip install transformers"
            ) from exc

        logger.info(
            "Loading generation model: %s",
            self.model_name,
        )

        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model_name,
        }

        if self.device is not None:
            pipeline_kwargs["device"] = self.device
        else:
            # Let Transformers choose an appropriate device.
            pipeline_kwargs["device_map"] = "auto"

        self._pipeline = pipeline(**pipeline_kwargs)

        logger.info("Generation model ready.")

    def generate(self, prompt: str) -> str:
        """Generate an answer from a complete RAG prompt.

        Args:
            prompt: Prompt produced by PromptBuilder.

        Returns:
            Generated answer with the original prompt removed.
        """
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        self._load()

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
        }

        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        else:
            generation_kwargs["do_sample"] = False

        # Some causal language models do not define a pad token.
        tokenizer = self._pipeline.tokenizer

        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

        try:
            outputs = self._pipeline(
                prompt,
                **generation_kwargs,
            )
        except Exception as exc:
            logger.exception("Generation failed.")
            raise RuntimeError(
                f"LLM generation failed: {exc}"
            ) from exc

        if not outputs:
            raise RuntimeError(
                "The generation model returned no output."
            )

        generated = outputs[0].get("generated_text")

        if not isinstance(generated, str):
            raise RuntimeError(
                "Unexpected generation output format."
            )

        return generated.strip()

    @property
    def is_loaded(self) -> bool:
        """Return whether the underlying model has been loaded."""
        return self._pipeline is not None
