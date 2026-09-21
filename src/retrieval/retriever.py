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

from collections.abc import Iterable
from dataclasses import dataclass

from exception import SensorException
from logger import logging
from src.config import Settings, get_settings
from src.models import RetrievalResult, TranscriptChunk

from .embeddings import EmbeddingService
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional metadata filters for source-grounded retrieval."""

    document_id: str | None = None
    expert_name: str | None = None
    market: str | None = None
    source_file: str | None = None


class Retriever:
    """Application-level source-grounded retrieval service."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
        top_k: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.embedding_service = (
            embedding_service
            or (
                vector_store.embedding_service
                if vector_store is not None
                else None
            )
        )

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = FAISSVectorStore(
                embedding_service=self.embedding_service,
                settings=self.settings,
            )

        self.top_k = (
            int(top_k)
            if top_k is not None
            else int(self.settings.retrieval_top_k)
        )

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant transcript chunks."""
        try:
            normalized_query = self._validate_query(query)

            requested_top_k = (
                int(top_k)
                if top_k is not None
                else self.top_k
            )

            if requested_top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

            results = self.vector_store.search(
                query=normalized_query,
                top_k=requested_top_k,
                min_score=min_score,
                market=(
                    filters.market
                    if filters is not None
                    else None
                ),
                expert_name=(
                    filters.expert_name
                    if filters is not None
                    else None
                ),
                document_id=(
                    filters.document_id
                    if filters is not None
                    else None
                ),
                source_file=(
                    filters.source_file
                    if filters is not None
                    else None
                ),
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
        """Retrieve evidence specifically from one expert."""
        if not expert_name or not expert_name.strip():
            raise ValueError(
                "expert_name cannot be empty."
            )

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

        The market helper intentionally searches the complete semantic
        candidate space for the requested market before applying the
        application's optional score threshold.

        This prevents a valid market-specific source from disappearing
        merely because a short query such as "adoption" has a lower
        embedding similarity to that market's wording.

        An explicitly supplied min_score is still respected.
        """
        if not market or not market.strip():
            raise ValueError(
                "market cannot be empty."
            )

        normalized_market = " ".join(
            market.split()
        ).casefold()

        requested_top_k = (
            int(top_k)
            if top_k is not None
            else self.top_k
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # If the caller explicitly supplies a score threshold, preserve
        # that contract. Otherwise use zero for this helper so that the
        # market filter is not defeated by the global retrieval threshold.
        effective_min_score = (
            float(min_score)
            if min_score is not None
            else 0.0
        )

        results = self.retrieve(
            query=query,
            top_k=requested_top_k,
            min_score=effective_min_score,
            filters=RetrievalFilters(
                market=market.strip(),
            ),
        )

        filtered_results = [
            result
            for result in results
            if result.market
            and " ".join(
                result.market.split()
            ).casefold()
            == normalized_market
        ]

        return self._reassign_ranks(
            filtered_results
        )

    def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve evidence from one transcript document."""
        if not document_id or not document_id.strip():
            raise ValueError(
                "document_id cannot be empty."
            )

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=RetrievalFilters(
                document_id=document_id.strip(),
            ),
        )

    def retrieve_for_source_file(
        self,
        query: str,
        source_file: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve evidence from one source transcript file."""
        if not source_file or not source_file.strip():
            raise ValueError(
                "source_file cannot be empty."
            )

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=RetrievalFilters(
                source_file=source_file.strip(),
            ),
        )

    def retrieve_for_all_experts(
        self,
        query: str,
        expert_names: Iterable[str] | None = None,
        top_k_per_expert: int | None = None,
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence across all requested experts.

        If expert_names is supplied, each expert is searched independently.
        Otherwise the entire corpus is searched.
        """
        try:
            if expert_names is None:
                return self.retrieve(
                    query=query,
                    top_k=(
                        top_k
                        if top_k is not None
                        else top_k_per_expert
                    ),
                    min_score=min_score,
                )

            normalized_names = self._normalize_expert_names(
                expert_names
            )

            if not normalized_names:
                return []

            per_expert_k = (
                int(top_k_per_expert)
                if top_k_per_expert is not None
                else self.top_k
            )

            if per_expert_k <= 0:
                raise ValueError(
                    "top_k_per_expert must be greater than zero."
                )

            merged_results: list[RetrievalResult] = []

            for expert_name in normalized_names:
                merged_results.extend(
                    self.retrieve_for_expert(
                        query=query,
                        expert_name=expert_name,
                        top_k=per_expert_k,
                        min_score=min_score,
                    )
                )

            merged_results = self._deduplicate_results(
                merged_results
            )

            merged_results.sort(
                key=lambda result: result.score,
                reverse=True,
            )

            if top_k is not None:
                if top_k <= 0:
                    raise ValueError(
                        "top_k must be greater than zero."
                    )

                merged_results = merged_results[:top_k]

            return self._reassign_ranks(
                merged_results
            )

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

    def retrieve_all_experts(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve evidence across the complete indexed corpus."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    def get_source_chunks(
        self,
        results: list[RetrievalResult],
    ) -> list[TranscriptChunk]:
        """Extract source chunks from retrieval results."""
        return [
            result.chunk
            for result in results
            if result.chunk is not None
        ]

    def has_sufficient_evidence(
        self,
        results: list[RetrievalResult],
        minimum_results: int = 1,
        minimum_score: float | None = None,
    ) -> bool:
        """Determine whether retrieval returned enough evidence."""
        if minimum_results <= 0:
            raise ValueError(
                "minimum_results must be greater than zero."
            )

        if not results:
            return False

        threshold = (
            float(minimum_score)
            if minimum_score is not None
            else float(
                self.settings.min_retrieval_score
            )
        )

        qualifying = [
            result
            for result in results
            if result.score >= threshold
        ]

        return len(qualifying) >= minimum_results

    @staticmethod
    def _deduplicate_results(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Remove duplicate chunks while preserving retrieval order."""
        seen: set[str] = set()
        unique_results: list[RetrievalResult] = []

        for result in results:
            chunk_id = result.chunk_id

            if not chunk_id and result.chunk is not None:
                chunk_id = result.chunk.chunk_id

            if not chunk_id:
                unique_results.append(result)
                continue

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            unique_results.append(result)

        return unique_results

    @staticmethod
    def _reassign_ranks(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Reassign ranks after filtering and deduplication."""
        return [
            result.model_copy(
                update={
                    "rank": rank,
                    "retrieval_rank": rank,
                }
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize a user query."""
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
    def _normalize_expert_names(
        expert_names: Iterable[str],
    ) -> list[str]:
        """Normalize and deduplicate expert names."""
        normalized: list[str] = []
        seen: set[str] = set()

        for name in expert_names:
            if not isinstance(name, str):
                continue

            cleaned = " ".join(
                name.split()
            )

            if not cleaned:
                continue

            key = cleaned.casefold()

            if key in seen:
                continue

            seen.add(key)
            normalized.append(cleaned)

        return normalized

    @staticmethod
    def _truncate_for_log(
        value: str,
        max_length: int = 120,
    ) -> str:
        """Keep query logging concise."""
        normalized = " ".join(
            value.split()
        )

        if len(normalized) <= max_length:
            return normalized

        return (
            f"{normalized[:max_length - 3]}..."
        )


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "RetrievalFilters",
    "Retriever",
]