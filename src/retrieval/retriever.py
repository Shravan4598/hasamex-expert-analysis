"""
Source-grounded retrieval service.

This module provides the application-level retrieval interface used by
the analysis layer.

Responsibilities:
    - Validate user queries.
    - Search the FAISS vector store.
    - Apply configurable result limits.
    - Support expert/market/document filters.
    - Deduplicate retrieved chunks.
    - Preserve source and timestamp metadata.
    - Provide useful diagnostics when evidence is insufficient.

The retriever does not generate answers. It only finds candidate source
evidence. Answer generation belongs to the analysis layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from exception import SensorException
from logger import logging

from src.config import Settings, get_settings
from src.models import RetrievalResult, TranscriptChunk

from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalFilters:
    """
    Optional metadata filters for source-grounded retrieval.

    Any field left as None is not used as a filter.
    """

    document_id: str | None = None
    expert_name: str | None = None
    market: str | None = None
    source_file: str | None = None


class Retriever:
    """
    Retrieve semantically relevant transcript chunks.

    The class deliberately keeps retrieval separate from reranking so
    each stage can be evaluated independently.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: Optional initialized FAISS vector store.
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.vector_store = vector_store or FAISSVectorStore(
            settings=self.settings,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant transcript chunks.

        Args:
            query: Natural-language user question.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score.
            filters: Optional metadata constraints.

        Returns:
            Source-traceable retrieval results sorted by similarity.

        Raises:
            SensorException: If retrieval fails.
        """
        try:
            normalized_query = self._validate_query(query)

            requested_top_k = (
                top_k
                if top_k is not None
                else self.settings.retrieval_top_k
            )

            if requested_top_k <= 0:
                raise ValueError("top_k must be greater than zero.")

            candidate_k = self._calculate_candidate_k(
                requested_top_k=requested_top_k,
                filters=filters,
            )

            results = self.vector_store.search(
                query=normalized_query,
                top_k=candidate_k,
                min_score=min_score,
            )

            if filters is not None:
                results = self._apply_filters(
                    results=results,
                    filters=filters,
                )

            results = self._deduplicate_results(results)
            results = results[:requested_top_k]

            results = self._reassign_ranks(results)

            logger.info(
                "Retrieved %d result(s) for query: %s",
                len(results),
                self._truncate_for_log(normalized_query),
            )

            return results

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Source retrieval failed.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def retrieve_for_expert(
        self,
        query: str,
        expert_name: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence specifically from one expert.

        Args:
            query: User question.
            expert_name: Expert name to constrain retrieval.
            top_k: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            Matching retrieval results.
        """
        if not expert_name or not expert_name.strip():
            raise ValueError("expert_name cannot be empty.")

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=RetrievalFilters(
                expert_name=expert_name.strip(),
            ),
        )

    def retrieve_for_market(
        self,
        query: str,
        market: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence specifically from one market.

        Args:
            query: User question.
            market: Market/country to constrain retrieval.
            top_k: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            Matching retrieval results.
        """
        if not market or not market.strip():
            raise ValueError("market cannot be empty.")

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=RetrievalFilters(
                market=market.strip(),
            ),
        )

    def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence from one transcript document.

        Args:
            query: User question.
            document_id: Stable transcript document identifier.
            top_k: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            Matching retrieval results.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=RetrievalFilters(
                document_id=document_id.strip(),
            ),
        )

    def retrieve_for_all_experts(
        self,
        query: str,
        expert_names: Iterable[str],
        top_k_per_expert: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """
        Retrieve evidence independently for multiple experts.

        This is useful for cross-expert analysis because it prevents a
        single expert with more semantically similar text from consuming
        all retrieval slots.

        Args:
            query: User question.
            expert_names: Expert names to search independently.
            top_k_per_expert: Maximum results for each expert.
            min_score: Minimum similarity score.

        Returns:
            Mapping from expert name to retrieval results.
        """
        try:
            normalized_names = self._normalize_expert_names(expert_names)

            if not normalized_names:
                return {}

            results: dict[str, list[RetrievalResult]] = {}

            for expert_name in normalized_names:
                results[expert_name] = self.retrieve_for_expert(
                    query=query,
                    expert_name=expert_name,
                    top_k=top_k_per_expert,
                    min_score=min_score,
                )

            return results

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to retrieve evidence for multiple experts."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def get_source_chunks(
        self,
        results: list[RetrievalResult],
    ) -> list[TranscriptChunk]:
        """
        Extract source chunks from retrieval results.

        Args:
            results: Retrieval results.

        Returns:
            TranscriptChunk objects in retrieval order.
        """
        return [result.chunk for result in results]

    def has_sufficient_evidence(
        self,
        results: list[RetrievalResult],
        minimum_results: int = 1,
        minimum_score: float | None = None,
    ) -> bool:
        """
        Determine whether retrieval returned enough evidence to proceed.

        This is a retrieval-level guard. The analysis layer may apply
        stricter evidence-coverage requirements before generating an answer.

        Args:
            results: Retrieved evidence.
            minimum_results: Minimum number of qualifying results.
            minimum_score: Optional score threshold.

        Returns:
            True if sufficient evidence exists.
        """
        if minimum_results <= 0:
            raise ValueError(
                "minimum_results must be greater than zero."
            )

        if not results:
            return False

        threshold = (
            minimum_score
            if minimum_score is not None
            else self.settings.min_retrieval_score
        )

        qualifying = [
            result
            for result in results
            if result.score >= threshold
        ]

        return len(qualifying) >= minimum_results

    def _calculate_candidate_k(
        self,
        requested_top_k: int,
        filters: RetrievalFilters | None,
    ) -> int:
        """
        Request additional FAISS candidates when metadata filtering is used.

        Filtering happens after semantic search because FAISS itself does
        not know the application-level TranscriptChunk metadata.
        """
        if filters is None:
            return requested_top_k

        multiplier = 4

        maximum_candidates = max(
            requested_top_k,
            min(
                self.vector_store.size,
                requested_top_k * multiplier,
            ),
        )

        return maximum_candidates

    @staticmethod
    def _apply_filters(
        results: list[RetrievalResult],
        filters: RetrievalFilters,
    ) -> list[RetrievalResult]:
        """
        Apply exact metadata filters to retrieval candidates.
        """
        filtered: list[RetrievalResult] = []

        for result in results:
            chunk = result.chunk

            if (
                filters.document_id is not None
                and chunk.document_id != filters.document_id
            ):
                continue

            if (
                filters.expert_name is not None
                and not Retriever._name_matches(
                    chunk.expert_name,
                    filters.expert_name,
                )
            ):
                continue

            if (
                filters.market is not None
                and not Retriever._value_matches(
                    chunk.market,
                    filters.market,
                )
            ):
                continue

            if (
                filters.source_file is not None
                and not Retriever._value_matches(
                    chunk.source_file,
                    filters.source_file,
                )
            ):
                continue

            filtered.append(result)

        return filtered

    @staticmethod
    def _deduplicate_results(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Remove duplicate chunks while preserving retrieval order.
        """
        seen: set[str] = set()
        unique_results: list[RetrievalResult] = []

        for result in results:
            chunk_id = result.chunk.chunk_id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            unique_results.append(result)

        return unique_results

    @staticmethod
    def _reassign_ranks(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Reassign ranks after filtering and deduplication.
        """
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(results, start=1)
        ]

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate and normalize a user query.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"Query must be a string, got {type(query).__name__}."
            )

        normalized = " ".join(query.split())

        if not normalized:
            raise ValueError("Query cannot be empty.")

        return normalized

    @staticmethod
    def _normalize_expert_names(
        expert_names: Iterable[str],
    ) -> list[str]:
        """
        Normalize and deduplicate expert names.
        """
        normalized: list[str] = []
        seen: set[str] = set()

        for name in expert_names:
            if not isinstance(name, str):
                continue

            cleaned = " ".join(name.split())

            if not cleaned:
                continue

            key = cleaned.casefold()

            if key in seen:
                continue

            seen.add(key)
            normalized.append(cleaned)

        return normalized

    @staticmethod
    def _name_matches(
        actual: str | None,
        requested: str,
    ) -> bool:
        """
        Match expert names conservatively.

        Exact normalized matching is preferred, with a controlled
        substring fallback for cases such as 'Jean Martin' versus
        'Dr. Jean Martin'.
        """
        if not actual:
            return False

        actual_normalized = Retriever._normalize_name(actual)
        requested_normalized = Retriever._normalize_name(requested)

        if actual_normalized == requested_normalized:
            return True

        return (
            requested_normalized in actual_normalized
            or actual_normalized in requested_normalized
        )

    @staticmethod
    def _value_matches(
        actual: str | None,
        requested: str,
    ) -> bool:
        """
        Perform case-insensitive exact matching for metadata values.
        """
        if actual is None:
            return False

        return (
            " ".join(actual.split()).casefold()
            == " ".join(requested.split()).casefold()
        )

    @staticmethod
    def _normalize_name(value: str) -> str:
        """
        Normalize an expert name for comparison.
        """
        normalized = value.casefold()
        normalized = normalized.replace("’", "'")
        normalized = normalized.replace(".", " ")

        if normalized.startswith("dr "):
            normalized = normalized[3:]

        return " ".join(normalized.split())

    @staticmethod
    def _truncate_for_log(
        value: str,
        max_length: int = 120,
    ) -> str:
        """
        Keep query logging concise.
        """
        normalized = " ".join(value.split())

        if len(normalized) <= max_length:
            return normalized

        return f"{normalized[:max_length - 3]}..."



def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "RetrievalFilters",
    "Retriever",
]