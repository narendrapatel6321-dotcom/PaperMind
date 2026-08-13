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

from src.retriever import RetrievedContext

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
        
    def _max_context_length(self) -> int | None:
        """Best-effort lookup of the model's max context window in tokens.

        Returns None if no reliable limit can be determined, in which case
        the length guard in generate() is skipped.
        """
        model_config = getattr(self._pipeline.model, "config", None)

        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return value

        tokenizer_limit = getattr(
            self._pipeline.tokenizer, "model_max_length", None
            )

        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit

        return None
            
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

        # Instruct-tuned models are fine-tuned on a chat template (system/
        # user/assistant turns). Use it when available so the model gets the
        # input format it was actually trained on; fall back to the raw
        # prompt string for base/non-chat models.
        use_chat_template = bool(getattr(tokenizer, "chat_template", None))

        if use_chat_template:
            model_input = [{"role": "user", "content": prompt}]
        else:
            model_input = prompt

        max_context = self._max_context_length()

        if max_context is not None:
            if use_chat_template:
                prompt_token_count = len(
                    tokenizer.apply_chat_template(
                        model_input,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            else:
                prompt_token_count = len(tokenizer.encode(prompt))

            available_for_prompt = max_context - self.max_new_tokens

            if prompt_token_count > available_for_prompt:
                raise ValueError(
                    "Prompt is too long for this model's context window: "
                    f"{prompt_token_count} prompt tokens + "
                    f"{self.max_new_tokens} max_new_tokens exceeds the "
                    f"model's limit of {max_context} tokens "
                    f"({prompt_token_count - available_for_prompt} tokens "
                    "over budget). Reduce top_k, chunk size, or "
                    "max_new_tokens."
                )

        try:
            outputs = self._pipeline(
                model_input,
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
