
"""
Second-stage retrieval reranking.

The initial FAISS retrieval is optimized for semantic similarity.
This module performs a lightweight deterministic reranking step using
lexical relevance signals.

Design goals:
    - Keep reranking deterministic and explainable.
    - Avoid introducing another model dependency.
    - Preserve source metadata.
    - Never manufacture evidence.
    - Return only chunks already retrieved by the vector store.

The reranker combines:
    1. Semantic retrieval score.
    2. Query-token overlap with the chunk.
    3. Exact phrase matching.
    4. Question-term coverage.
    5. Optional expert diversity.

The resulting score is used only for ordering.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from exception import SensorException
from logger import logging
from src.config import Settings, get_settings
from src.models import RetrievalResult

logger = logging.getLogger(__name__)


class RerankConfig:
    """Configuration for deterministic retrieval reranking."""

    def __init__(
        self,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.20,
        phrase_weight: float = 0.10,
        coverage_weight: float = 0.05,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.phrase_weight = phrase_weight
        self.coverage_weight = coverage_weight

        weights = (
            semantic_weight,
            lexical_weight,
            phrase_weight,
            coverage_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "Reranking weights cannot be negative."
            )

        if not math.isclose(
            sum(weights),
            1.0,
            rel_tol=1e-9,
        ):
            raise ValueError(
                "Reranking weights must sum to 1.0."
            )


class RetrievalReranker:
    """
    Deterministically rerank candidate retrieval results.

    This component operates only on candidates already returned by
    the retriever.
    """

    STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "there",
            "these",
            "this",
            "to",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
    )

    TOKEN_PATTERN = re.compile(
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*"
    )

    def __init__(
        self,
        settings: Settings | None = None,
        config: RerankConfig | None = None,
    ) -> None:
        """Initialize the reranker."""

        self.settings = settings or get_settings()
        self.config = config or RerankConfig()

    # ==================================================================
    # NORMAL RERANKING
    # ==================================================================

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval candidates against the original query.
        """

        try:
            normalized_query = self._validate_query(query)

            candidates = list(results)

            if not candidates:
                return []

            requested_top_k = (
                top_k
                if top_k is not None
                else self.settings.rerank_top_k
            )

            if requested_top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

            query_tokens = self._meaningful_tokens(
                normalized_query
            )

            scored_results: list[
                tuple[float, RetrievalResult]
            ] = []

            for result in candidates:
                score = self._calculate_score(
                    query=normalized_query,
                    query_tokens=query_tokens,
                    result=result,
                )

                scored_results.append(
                    (
                        score,
                        result,
                    )
                )

            scored_results.sort(
                key=lambda item: (
                    item[0],
                    item[1].score,
                ),
                reverse=True,
            )

            reranked: list[RetrievalResult] = []

            for rank, (
                rerank_score,
                result,
            ) in enumerate(
                scored_results[:requested_top_k],
                start=1,
            ):
                updated_result = result.model_copy(
                    update={
                        "rank": rank,
                        "score": rerank_score,
                    }
                )

                reranked.append(
                    updated_result
                )

            logger.info(
                "Reranked %d candidate(s) to %d result(s) for query: %s",
                len(candidates),
                len(reranked),
                self._truncate_for_log(
                    normalized_query
                ),
            )

            return reranked

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to rerank retrieval results."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ==================================================================
    # EXPERT-BALANCED RERANKING
    # ==================================================================

    def rerank_by_expert_balance(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        expert_names: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank results while encouraging representation across experts.

        IMPORTANT:
            expert_names is intentionally optional.

        If expert_names is supplied:
            the method encourages representation from different experts.

        If expert_names is None or empty:
            the method simply performs normal reranking.

        This makes the method compatible with callers that do not
        have an explicit expert scope.
        """

        try:
            # ----------------------------------------------------------
            # Always start with deterministic relevance reranking.
            # ----------------------------------------------------------

            base_results = self.rerank(
                query=query,
                results=results,
                top_k=None,
            )

            if not base_results:
                return []

            # ----------------------------------------------------------
            # Normalize expert names safely.
            # ----------------------------------------------------------

            normalized_experts: set[str] = set()

            if expert_names is not None:
                normalized_experts = {
                    self._normalize_name(name)
                    for name in expert_names
                    if isinstance(name, str)
                    and name.strip()
                }

            # ----------------------------------------------------------
            # If no expert information exists, use normal reranking.
            # ----------------------------------------------------------

            if not normalized_experts:
                target_k = (
                    top_k
                    if top_k is not None
                    else self.settings.rerank_top_k
                )

                return self._reassign_ranks(
                    base_results[:target_k]
                )

            # ----------------------------------------------------------
            # Determine target result count.
            # ----------------------------------------------------------

            target_k = (
                top_k
                if top_k is not None
                else self.settings.rerank_top_k
            )

            if target_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

            # ----------------------------------------------------------
            # Expert-balanced selection.
            # ----------------------------------------------------------

            selected: list[RetrievalResult] = []

            remaining = list(
                base_results
            )

            represented: set[str] = set()

            # ----------------------------------------------------------
            # First pass:
            # Select the strongest result from each unrepresented
            # expert where possible.
            # ----------------------------------------------------------

            while (
                remaining
                and len(selected) < target_k
            ):
                best_index = (
                    self._find_best_unrepresented_expert(
                        remaining=remaining,
                        represented=represented,
                        expected_experts=normalized_experts,
                    )
                )

                if best_index is None:
                    break

                result = remaining.pop(
                    best_index
                )

                selected.append(
                    result
                )

                expert_key = self._normalize_name(
                    result.chunk.expert_name
                )

                if expert_key:
                    represented.add(
                        expert_key
                    )

            # ----------------------------------------------------------
            # Second pass:
            # Fill remaining slots according to relevance.
            # ----------------------------------------------------------

            for result in remaining:
                if len(selected) >= target_k:
                    break

                selected.append(
                    result
                )

            logger.info(
                "Applied expert-balanced reranking: "
                "%d selected result(s), %d expert(s) represented.",
                len(selected),
                len(represented),
            )

            return self._reassign_ranks(
                selected
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to apply expert-balanced reranking."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ==================================================================
    # SCORE EXPLANATION
    # ==================================================================

    def explain_score(
        self,
        query: str,
        result: RetrievalResult,
    ) -> dict[str, float]:
        """
        Explain the deterministic reranking score.
        """

        normalized_query = self._validate_query(
            query
        )

        query_tokens = self._meaningful_tokens(
            normalized_query
        )

        semantic_score = (
            self._normalize_semantic_score(
                result.score
            )
        )

        lexical_score = self._lexical_overlap(
            query_tokens=query_tokens,
            chunk_text=result.chunk.text,
        )

        phrase_score = self._phrase_match(
            query=normalized_query,
            chunk_text=result.chunk.text,
        )

        coverage_score = self._term_coverage(
            query_tokens=query_tokens,
            chunk_text=result.chunk.text,
        )

        final_score = (
            self.config.semantic_weight
            * semantic_score
            + self.config.lexical_weight
            * lexical_score
            + self.config.phrase_weight
            * phrase_score
            + self.config.coverage_weight
            * coverage_score
        )

        return {
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "phrase_score": phrase_score,
            "coverage_score": coverage_score,
            "final_score": final_score,
        }

    def _calculate_score(
        self,
        query: str,
        query_tokens: set[str],
        result: RetrievalResult,
    ) -> float:
        """Calculate deterministic reranking score."""

        components = self.explain_score(
            query=query,
            result=result,
        )

        return components["final_score"]

    # ==================================================================
    # LEXICAL SCORING
    # ==================================================================

    def _lexical_overlap(
        self,
        query_tokens: set[str],
        chunk_text: str,
    ) -> float:
        """Calculate Jaccard-style lexical overlap."""

        if not query_tokens:
            return 0.0

        chunk_tokens = self._meaningful_tokens(
            chunk_text
        )

        if not chunk_tokens:
            return 0.0

        intersection = (
            query_tokens.intersection(
                chunk_tokens
            )
        )

        union = (
            query_tokens.union(
                chunk_tokens
            )
        )

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _term_coverage(
        self,
        query_tokens: set[str],
        chunk_text: str,
    ) -> float:
        """Calculate query-term coverage."""

        if not query_tokens:
            return 0.0

        chunk_tokens = self._meaningful_tokens(
            chunk_text
        )

        matched = (
            query_tokens.intersection(
                chunk_tokens
            )
        )

        return len(matched) / len(query_tokens)

    def _phrase_match(
        self,
        query: str,
        chunk_text: str,
    ) -> float:
        """Calculate phrase matching score."""

        normalized_query = (
            self._normalize_for_matching(
                query
            )
        )

        normalized_chunk = (
            self._normalize_for_matching(
                chunk_text
            )
        )

        if (
            not normalized_query
            or not normalized_chunk
        ):
            return 0.0

        if normalized_query in normalized_chunk:
            return 1.0

        query_tokens = (
            normalized_query.split()
        )

        if len(query_tokens) < 2:
            return 0.0

        chunk_tokens = (
            normalized_chunk.split()
        )

        if len(chunk_tokens) < 2:
            return 0.0

        query_bigrams = {
            f"{query_tokens[index]} "
            f"{query_tokens[index + 1]}"
            for index in range(
                len(query_tokens) - 1
            )
        }

        chunk_bigrams = {
            f"{chunk_tokens[index]} "
            f"{chunk_tokens[index + 1]}"
            for index in range(
                len(chunk_tokens) - 1
            )
        }

        if not query_bigrams:
            return 0.0

        return len(
            query_bigrams.intersection(
                chunk_bigrams
            )
        ) / len(query_bigrams)

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    def _normalize_semantic_score(
        self,
        score: float,
    ) -> float:
        """
        Convert cosine similarity into [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                (
                    float(score)
                    + 1.0
                )
                / 2.0,
            ),
        )

    def _meaningful_tokens(
        self,
        text: str,
    ) -> set[str]:
        """Extract lowercase meaningful tokens."""

        if not isinstance(text, str):
            return set()

        tokens = self.TOKEN_PATTERN.findall(
            text.casefold()
        )

        return {
            token
            for token in tokens
            if token not in self.STOPWORDS
            and len(token) > 1
        }

    @staticmethod
    def _normalize_for_matching(
        text: str,
    ) -> str:
        """Normalize text for phrase matching."""

        if not isinstance(text, str):
            return ""

        text = text.casefold()

        text = text.replace(
            "’",
            "'",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"[^\w\s'-]",
            "",
            text,
        )

        return text.strip()

    # ==================================================================
    # EXPERT BALANCE HELPERS
    # ==================================================================

    def _find_best_unrepresented_expert(
        self,
        remaining: list[RetrievalResult],
        represented: set[str],
        expected_experts: set[str],
    ) -> int | None:
        """
        Find strongest result belonging to an unrepresented expert.
        """

        best_index: int | None = None
        best_score = float("-inf")

        for index, result in enumerate(
            remaining
        ):
            expert = self._normalize_name(
                result.chunk.expert_name
            )

            if not expert:
                continue

            if expert in represented:
                continue

            # Match exact names first, then tolerate small metadata
            # differences such as "Dr John Smith" vs "John Smith".
            if expected_experts:
                matched = any(
                    expected == expert
                    or expected in expert
                    or expert in expected
                    for expected in expected_experts
                )

                if not matched:
                    continue

            if result.score > best_score:
                best_score = result.score
                best_index = index

        return best_index

    @staticmethod
    def _reassign_ranks(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Reassign ranks after diversity processing."""

        return [
            result.model_copy(
                update={
                    "rank": rank
                }
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

    @staticmethod
    def _normalize_name(
        value: str | None,
    ) -> str:
        """Normalize expert names for comparison."""

        if not value:
            return ""

        normalized = value.casefold()

        normalized = normalized.replace(
            "’",
            "'",
        )

        normalized = normalized.replace(
            ".",
            " ",
        )

        normalized = normalized.removeprefix(
            "dr "
        )

        return " ".join(
            normalized.split()
        )

    # ==================================================================
    # VALIDATION / LOGGING
    # ==================================================================

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:
        """Validate and normalize a query."""

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string, "
                f"got {type(query).__name__}."
            )

        normalized = " ".join(
            query.split()
        )

        if not normalized:
            raise ValueError(
                "Query cannot be empty."
            )

        return normalized

    @staticmethod
    def _truncate_for_log(
        value: str,
        max_length: int = 120,
    ) -> str:
        """Keep query text concise in logs."""

        normalized = " ".join(
            value.split()
        )

        if len(normalized) <= max_length:
            return normalized

        return (
            f"{normalized[:max_length - 3]}..."
        )


def _sys_module():
    """Return the active sys module."""

    import sys

    return sys


__all__ = [
    "RerankConfig",
    "RetrievalReranker",
]

