"""Evaluation utilities for the PaperMind RAG pipeline.

This module provides two levels of evaluation:

1. Retrieval evaluation
   - Hit@K
   - Mean Reciprocal Rank (MRR)
   - Context relevance
   - Source diversity

2. Answer evaluation
   - Faithfulness
   - Token-level precision / recall / F1

The retrieval metrics are the primary benchmark metrics for comparing
embedding models and retriever configurations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from src.retriever import RetrievedContext

logger = logging.getLogger(__name__)

# Text utilities
_STOPWORDS = frozenset(
    {
        "what",
        "is",
        "the",
        "a",
        "an",
        "how",
        "does",
        "do",
        "in",
        "of",
        "for",
        "and",
        "or",
        "to",
        "that",
        "this",
        "it",
        "are",
        "was",
        "were",
        "be",
        "has",
        "have",
        "had",
        "with",
        "by",
        "from",
        "on",
        "at",
        "as",
        "its",
        "their",
        "which",
        "who",
        "why",
        "where",
        "when",
    }
)

def _content_words(text: str) -> set[str]:
    """Return lowercase content words from text."""
    tokens = set(re.findall(r"\b[a-z]{2,}\b", text.lower()))
    return tokens - _STOPWORDS


def _normalize_answer(text: str) -> list[str]:
    """Tokenize text for answer-level evaluation."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

# Retrieval result
@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics for an evaluation set."""

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    num_questions: int

    def as_dict(self) -> dict[str, float | int]:
        """Return metrics as a serializable dictionary."""
        return {
            "hit@1": self.hit_at_1,
            "hit@3": self.hit_at_3,
            "hit@5": self.hit_at_5,
            "hit@10": self.hit_at_10,
            "mrr": self.mrr,
            "num_questions": self.num_questions,
        }


# Retrieval evaluator
class RetrievalEvaluator:
    """Evaluate retrieval quality against manually labelled ground truth."""

    @staticmethod
    def _chunk_key(chunk) -> tuple[str, int]:
        """Return a stable identifier for a retrieved chunk."""
        return chunk.source, chunk.chunk_id

    def hit_at_k(
        self,
        retrieved: RetrievedContext,
        relevant_chunks: Iterable[tuple[str, int]],
        k: int,
    ) -> float:
        """Return 1 if a relevant chunk appears in the top-k results."""
        if k <= 0:
            raise ValueError("k must be positive")

        relevant = set(relevant_chunks)
        retrieved_keys = {
            self._chunk_key(chunk)
            for chunk in retrieved.chunks[:k]
        }

        return float(bool(retrieved_keys & relevant))

    def reciprocal_rank(
        self,
        retrieved: RetrievedContext,
        relevant_chunks: Iterable[tuple[str, int]],
    ) -> float:
        """Return reciprocal rank of the first relevant chunk.

        Returns 0.0 when no relevant chunk is retrieved.
        """
        relevant = set(relevant_chunks)

        for rank, chunk in enumerate(retrieved.chunks, start=1):
            if self._chunk_key(chunk) in relevant:
                return 1.0 / rank

        return 0.0

    def evaluate_query(
        self,
        retrieved: RetrievedContext,
        relevant_chunks: Iterable[tuple[str, int]],
    ) -> dict[str, float]:
        """Evaluate one query against its ground-truth chunks."""
        relevant = set(relevant_chunks)

        return {
            "hit@1": self.hit_at_k(retrieved, relevant, 1),
            "hit@3": self.hit_at_k(retrieved, relevant, 3),
            "hit@5": self.hit_at_k(retrieved, relevant, 5),
            "hit@10": self.hit_at_k(retrieved, relevant, 10),
            "rr": self.reciprocal_rank(retrieved, relevant),
        }

    def evaluate_dataset(
        self,
        results: dict[str, RetrievedContext],
        ground_truth: dict[str, Iterable[tuple[str, int]]],
    ) -> RetrievalMetrics:
        """Evaluate all queries and return aggregate retrieval metrics.

        Args:
            results:
                Mapping from question ID to RetrievedContext.

            ground_truth:
                Mapping from question ID to relevant
                ``(source, chunk_id)`` pairs.

        Returns:
            RetrievalMetrics containing Hit@1/3/5/10 and MRR.
        """
        if not results:
            raise ValueError("No retrieval results supplied.")

        query_metrics: list[dict[str, float]] = []

        for question_id, context in results.items():
            if question_id not in ground_truth:
                logger.warning(
                    "No ground truth found for question '%s'; skipping.",
                    question_id,
                )
                continue

            metrics = self.evaluate_query(
                context,
                ground_truth[question_id],
            )
            query_metrics.append(metrics)

        if not query_metrics:
            raise ValueError("No queries had matching ground truth.")

        n = len(query_metrics)

        return RetrievalMetrics(
            hit_at_1=sum(m["hit@1"] for m in query_metrics) / n,
            hit_at_3=sum(m["hit@3"] for m in query_metrics) / n,
            hit_at_5=sum(m["hit@5"] for m in query_metrics) / n,
            hit_at_10=sum(m["hit@10"] for m in query_metrics) / n,
            mrr=sum(m["rr"] for m in query_metrics) / n,
            num_questions=n,
        )

    def context_relevance(
        self,
        query: str,
        context: RetrievedContext,
    ) -> float:
        """Measure query keyword coverage in retrieved context.

        This is a diagnostic metric, not the primary retrieval benchmark.
        """
        keywords = _content_words(query)

        if not keywords:
            return 0.0

        full_context = " ".join(
            chunk.text.lower()
            for chunk in context.chunks
        )

        found = sum(
            1
            for keyword in keywords
            if keyword in full_context
        )

        return round(found / len(keywords), 4)

    def source_diversity(
        self,
        context: RetrievedContext,
    ) -> float:
        """Measure how diverse the retrieved source documents are."""
        if not context.chunks:
            return 0.0

        unique_sources = {
            chunk.source
            for chunk in context.chunks
        }

        return round(
            len(unique_sources) / len(context.chunks),
            4,
        )

# Answer evaluator
class AnswerEvaluator:
    """Evaluate generated answers against retrieved context or references."""

    def faithfulness(
        self,
        answer: str,
        context: RetrievedContext,
    ) -> float:
        """Measure how much answer vocabulary appears in the context.

        This is a lightweight lexical diagnostic. It should not be interpreted
        as a perfect semantic faithfulness metric.
        """
        answer_words = _content_words(answer)

        if not answer_words:
            return 0.0

        full_context = " ".join(
            chunk.text.lower()
            for chunk in context.chunks
        )

        found = sum(
            1
            for word in answer_words
            if word in full_context
        )

        return round(found / len(answer_words), 4)

    def token_f1(
        self,
        prediction: str,
        reference: str,
    ) -> dict[str, float]:
        """Compute token-level precision, recall, and F1.

        Tokens are normalized to lowercase alphanumeric strings.

        Returns:
            Dictionary containing ``precision``, ``recall`` and ``f1``.
        """
        prediction_tokens = _normalize_answer(prediction)
        reference_tokens = _normalize_answer(reference)

        if not prediction_tokens or not reference_tokens:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        prediction_counts: dict[str, int] = {}
        reference_counts: dict[str, int] = {}

        for token in prediction_tokens:
            prediction_counts[token] = (
                prediction_counts.get(token, 0) + 1
            )

        for token in reference_tokens:
            reference_counts[token] = (
                reference_counts.get(token, 0) + 1
            )

        common = 0

        for token, count in prediction_counts.items():
            common += min(
                count,
                reference_counts.get(token, 0),
            )

        if common == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        precision = common / len(prediction_tokens)
        recall = common / len(reference_tokens)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
