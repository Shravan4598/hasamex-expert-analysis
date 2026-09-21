"""
FAISS vector store for Hasamex Expert Analysis.

Responsibilities
----------------
- Store normalized transcript embeddings in FAISS.
- Preserve transcript metadata separately from vectors.
- Support both production chunk-based indexing and test/in-memory
  embedding-based indexing.
- Persist and reload the vector store.
- Provide deterministic similarity retrieval.
- Preserve source, expert, market, document, timestamp, and chunk identity.
- Support metadata filtering without modifying source chunks.

The implementation uses FAISS IndexFlatIP. Since the configured
EmbeddingService returns normalized embeddings, inner product is
equivalent to cosine similarity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from exception import SensorException
from logger import logging
from src.config import Settings, get_settings
from src.models import RetrievalResult, TranscriptChunk

from .embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Persistent FAISS vector store.

    The class intentionally supports two build styles:

    Production style
        store.build(chunks=[...])

    Low-level/test style
        store.build(
            embeddings=embeddings,
            metadata=[...],
        )

    This keeps the vector-store boundary reusable while ensuring that
    application metadata remains source traceable.
    """

    INDEX_FILENAME = "transcript.index"
    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
        storage_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the vector store.

        Parameters
        ----------
        embedding_service:
            Optional embedding service.

        settings:
            Optional application settings.

        storage_dir:
            Optional explicit storage directory.

            This is intentionally supported because tests and temporary
            application instances need isolated vector stores.
        """

        self.settings = settings or get_settings()

        self.embedding_service = (
            embedding_service
            or get_embedding_service()
        )

        if storage_dir is not None:
            self.storage_path = Path(storage_dir)
        else:
            self.storage_path = self.settings.vector_store_path

        self.index_path = (
            self.storage_path
            / self.INDEX_FILENAME
        )

        self.metadata_path = (
            self.storage_path
            / self.METADATA_FILENAME
        )

        self._index: faiss.Index | None = None

        self._chunks: list[TranscriptChunk] = []

    # ==================================================================
    # BASIC PROPERTIES
    # ==================================================================

    @property
    def index(self) -> faiss.Index | None:
        """Return the active FAISS index."""

        return self._index

    @property
    def chunks(self) -> list[TranscriptChunk]:
        """
        Return indexed transcript chunks.

        A shallow list copy is returned so callers cannot mutate the
        vector-store list itself.
        """

        return list(self._chunks)

    @property
    def count(self) -> int:
        """Return the number of indexed vectors."""

        if self._index is None:
            return 0

        return int(self._index.ntotal)

    @property
    def size(self) -> int:
        """Backward-compatible alias for count."""

        return self.count

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""

        if self._index is None:
            return 0

        return int(self._index.d)

    @property
    def is_ready(self) -> bool:
        """Return whether the index and metadata are synchronized."""

        return (
            self._index is not None
            and self._index.ntotal > 0
            and len(self._chunks) == self._index.ntotal
        )

    # ==================================================================
    # BUILD
    # ==================================================================

    def build(
        self,
        chunks: Sequence[TranscriptChunk] | None = None,
        persist: bool = True,
        embeddings: np.ndarray | Sequence[Sequence[float]] | None = None,
        metadata: Sequence[
            TranscriptChunk | Mapping[str, Any]
        ] | None = None,
    ) -> None:
        """
        Build a new FAISS index.

        Supported calling conventions:

        1. Production:

            build(chunks=[...])

        2. Tests / precomputed embeddings:

            build(
                embeddings=embeddings,
                metadata=[chunk.model_dump(...) for chunk in chunks],
            )

        Existing index contents are replaced.
        """

        try:
            resolved_chunks = self._resolve_chunks(
                chunks=chunks,
                metadata=metadata,
            )

            if not resolved_chunks:
                raise ValueError(
                    "Cannot build vector store from an empty chunk list."
                )

            # ----------------------------------------------------------
            # Use supplied embeddings when available.
            # Otherwise generate them from chunk text.
            # ----------------------------------------------------------

            if embeddings is None:
                texts = [
                    chunk.text
                    for chunk in resolved_chunks
                ]

                logger.info(
                    "Generating embeddings for %d transcript chunks.",
                    len(resolved_chunks),
                )

                embedding_matrix = (
                    self.embedding_service.embed_documents(
                        texts
                    )
                )
            else:
                embedding_matrix = np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

            self._validate_embeddings(
                embedding_matrix,
                expected_count=len(resolved_chunks),
            )

            # ----------------------------------------------------------
            # FAISS expects contiguous float32 arrays.
            # ----------------------------------------------------------

            embedding_matrix = np.ascontiguousarray(
                embedding_matrix,
                dtype=np.float32,
            )

            dimension = int(
                embedding_matrix.shape[1]
            )

            index = faiss.IndexFlatIP(
                dimension
            )

            index.add(
                embedding_matrix
            )

            self._index = index

            # Deep-copy metadata so retrieval cannot mutate the source
            # objects supplied by callers.
            self._chunks = [
                deepcopy(chunk)
                for chunk in resolved_chunks
            ]

            logger.info(
                "FAISS index built successfully: "
                "%d vectors, dimension=%d",
                index.ntotal,
                dimension,
            )

            if persist:
                self.save()

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to build FAISS vector store."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def add(
        self,
        chunks: Sequence[TranscriptChunk],
        persist: bool = True,
    ) -> None:
        """
        Add transcript chunks to an existing index.

        If no index exists, build() is used.
        """

        try:
            if not chunks:
                logger.warning(
                    "No chunks supplied to vector store add()."
                )
                return

            if self._index is None:
                self.build(
                    chunks=list(chunks),
                    persist=persist,
                )
                return

            texts = [
                chunk.text
                for chunk in chunks
            ]

            embeddings = (
                self.embedding_service.embed_documents(
                    texts
                )
            )

            self._validate_embeddings(
                embeddings,
                expected_count=len(chunks),
            )

            if embeddings.shape[1] != self.dimension:
                raise ValueError(
                    "Embedding dimension does not match "
                    f"existing FAISS index. Expected "
                    f"{self.dimension}, got "
                    f"{embeddings.shape[1]}."
                )

            self._index.add(
                np.ascontiguousarray(
                    embeddings,
                    dtype=np.float32,
                )
            )

            self._chunks.extend(
                deepcopy(list(chunks))
            )

            logger.info(
                "Added %d chunks. New vector count: %d",
                len(chunks),
                self.count,
            )

            if persist:
                self.save()

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to add chunks to FAISS vector store."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        market: str | None = None,
        expert_name: str | None = None,
        document_id: str | None = None,
        source_file: str | None = None,
        market_filter: str | None = None,
        expert_filter: str | None = None,
        **filters: Any,
    ) -> list[RetrievalResult]:
        """
        Search the vector store.

        Filters are applied against source metadata after retrieving a
        sufficiently large FAISS candidate pool. This is important:
        filtering only the first `top_k` FAISS results could accidentally
        remove all relevant candidates.

        Parameters
        ----------
        query:
            Natural-language search query.

        top_k:
            Maximum number of final results.

        min_score:
            Minimum similarity score.

        market / market_filter:
            Optional market filter.

        expert_name / expert_filter:
            Optional expert filter.

        document_id:
            Optional document filter.

        source_file:
            Optional source-file filter.
        """

        try:
            if not query or not query.strip():
                raise ValueError(
                    "Search query cannot be empty."
                )

            if not self.is_ready:
                raise RuntimeError(
                    "Vector store is not ready. "
                    "Build or load an index first."
                )

            requested_k = (
                top_k
                if top_k is not None
                else self.settings.retrieval_top_k
            )

            requested_k = int(requested_k)

            if requested_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

            threshold = (
                min_score
                if min_score is not None
                else self.settings.min_retrieval_score
            )

            threshold = float(threshold)

            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "min_score must be between 0.0 and 1.0."
                )

            # Support both explicit parameters and aliases.
            market_value = (
                market
                if market is not None
                else market_filter
            )

            expert_value = (
                expert_name
                if expert_name is not None
                else expert_filter
            )

            document_value = (
                document_id
                if document_id is not None
                else filters.get("document")
            )

            source_value = (
                source_file
                if source_file is not None
                else filters.get("source")
            )

            query_embedding = (
                self.embedding_service.embed_query(
                    query
                )
            )

            query_vector = np.asarray(
                query_embedding,
                dtype=np.float32,
            ).reshape(1, -1)

            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    "Query embedding dimension does not "
                    "match FAISS index dimension."
                )

            # ----------------------------------------------------------
            # Retrieve every vector when filters are present.
            #
            # This prevents filtering from eliminating relevant results
            # simply because they were below an unfiltered top-k cutoff.
            # ----------------------------------------------------------

            filters_active = any(
                value is not None
                for value in (
                    market_value,
                    expert_value,
                    document_value,
                    source_value,
                )
            )

            if filters_active:
                search_k = self.count
            else:
                search_k = min(
                    requested_k,
                    self.count,
                )

            scores, indices = self._index.search(
                np.ascontiguousarray(
                    query_vector,
                    dtype=np.float32,
                ),
                search_k,
            )

            results: list[RetrievalResult] = []

            for raw_score, raw_index in zip(
                scores[0],
                indices[0],
            ):
                index_position = int(raw_index)

                if index_position < 0:
                    continue

                if index_position >= len(self._chunks):
                    logger.warning(
                        "FAISS returned invalid metadata index: %d",
                        index_position,
                    )
                    continue

                similarity = float(raw_score)

                if similarity < threshold:
                    continue

                chunk = self._chunks[
                    index_position
                ]

                if not self._matches_filters(
                    chunk=chunk,
                    market=market_value,
                    expert_name=expert_value,
                    document_id=document_value,
                    source_file=source_value,
                ):
                    continue

                results.append(
                    self._to_retrieval_result(
                        chunk=chunk,
                        score=similarity,
                        rank=len(results) + 1,
                    )
                )

                if len(results) >= requested_k:
                    break

            logger.info(
                "Vector search returned %d result(s) for query: %s",
                len(results),
                self._truncate_for_log(query),
            )

            return results

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "FAISS vector search failed."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ==================================================================
    # RETRIEVAL HELPERS
    # ==================================================================

    def retrieve_for_expert(
        self,
        query: str,
        expert_name: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve evidence restricted to one expert."""

        return self.search(
            query=query,
            top_k=top_k,
            expert_name=expert_name,
            **kwargs,
        )

    def retrieve_for_market(
        self,
        query: str,
        market: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve evidence restricted to one market."""

        return self.search(
            query=query,
            top_k=top_k,
            market=market,
            **kwargs,
        )

    def retrieve_for_all_experts(
        self,
        query: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """
        Retrieve across all experts.

        No expert filter is applied.
        """

        return self.search(
            query=query,
            top_k=top_k,
            **kwargs,
        )

    # Backward-compatible naming.
    def retrieve_all_experts(
        self,
        query: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Alias for retrieve_for_all_experts."""

        return self.retrieve_for_all_experts(
            query=query,
            top_k=top_k,
            **kwargs,
        )

    # ==================================================================
    # PERSISTENCE
    # ==================================================================

    def save(self) -> None:
        """Persist FAISS index and source metadata."""

        try:
            if not self.is_ready:
                raise RuntimeError(
                    "Cannot save an empty or incomplete vector store."
                )

            self.storage_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            faiss.write_index(
                self._index,
                str(self.index_path),
            )

            metadata = {
                "version": 2,
                "embedding_model": (
                    self.embedding_service.model_name
                ),
                "embedding_dimension": self.dimension,
                "vector_count": self.count,
                "chunks": [
                    chunk.model_dump(
                        mode="json"
                    )
                    for chunk in self._chunks
                ],
            }

            self.metadata_path.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "Vector store persisted successfully: %s",
                self.storage_path,
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to persist vector store."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def load(self) -> None:
        """Load a previously persisted vector store."""

        try:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found: "
                    f"{self.index_path}"
                )

            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Vector metadata not found: "
                    f"{self.metadata_path}"
                )

            index = faiss.read_index(
                str(self.index_path)
            )

            metadata = json.loads(
                self.metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            self._validate_loaded_metadata(
                metadata=metadata,
                index=index,
            )

            raw_chunks = metadata.get(
                "chunks",
                [],
            )

            chunks = [
                TranscriptChunk.model_validate(
                    item
                )
                for item in raw_chunks
            ]

            if len(chunks) != index.ntotal:
                raise ValueError(
                    "Vector index and metadata count mismatch. "
                    f"Index contains {index.ntotal} vectors but "
                    f"metadata contains {len(chunks)} chunks."
                )

            self._index = index
            self._chunks = chunks

            logger.info(
                "Vector store loaded successfully: %d vectors.",
                index.ntotal,
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to load vector store."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def exists_on_disk(self) -> bool:
        """Return whether a complete persisted store exists."""

        return (
            self.index_path.is_file()
            and self.metadata_path.is_file()
        )

    def clear(self) -> None:
        """Clear only the in-memory store."""

        self._index = None
        self._chunks = []

        logger.info(
            "In-memory vector store cleared."
        )

    def delete_persisted_store(self) -> None:
        """Delete persisted FAISS and metadata files."""

        try:
            for path in (
                self.index_path,
                self.metadata_path,
            ):
                if path.exists():
                    path.unlink()

            self.clear()

            logger.info(
                "Persisted vector store deleted: %s",
                self.storage_path,
            )

        except Exception as error:
            logger.exception(
                "Failed to delete persisted vector store."
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================

    @staticmethod
    def _resolve_chunks(
        chunks: Sequence[TranscriptChunk] | None,
        metadata: Sequence[
            TranscriptChunk | Mapping[str, Any]
        ] | None,
    ) -> list[TranscriptChunk]:
        """
        Resolve metadata into strongly typed TranscriptChunk objects.
        """

        if chunks is not None:
            resolved = list(chunks)

            for item in resolved:
                if not isinstance(
                    item,
                    TranscriptChunk,
                ):
                    raise TypeError(
                        "All chunks must be TranscriptChunk instances."
                    )

            return resolved

        if metadata is None:
            raise ValueError(
                "Either chunks or metadata must be provided."
            )

        resolved_chunks: list[TranscriptChunk] = []

        for item in metadata:
            if isinstance(
                item,
                TranscriptChunk,
            ):
                resolved_chunks.append(
                    item
                )
            elif isinstance(
                item,
                Mapping,
            ):
                resolved_chunks.append(
                    TranscriptChunk.model_validate(
                        dict(item)
                    )
                )
            else:
                raise TypeError(
                    "Metadata items must be TranscriptChunk "
                    "objects or mappings."
                )

        return resolved_chunks

    @staticmethod
    def _matches_filters(
        chunk: TranscriptChunk,
        market: str | None = None,
        expert_name: str | None = None,
        document_id: str | None = None,
        source_file: str | None = None,
    ) -> bool:
        """Return whether a chunk satisfies every active filter."""

        if (
            market is not None
            and chunk.market != market
        ):
            return False

        if (
            expert_name is not None
            and chunk.expert_name != expert_name
        ):
            return False

        if (
            document_id is not None
            and chunk.document_id != document_id
        ):
            return False

        return not (
                source_file is not None
                and chunk.source_file != source_file)

    @staticmethod
    def _to_retrieval_result(
        chunk: TranscriptChunk,
        score: float,
        rank: int,
    ) -> RetrievalResult:
        """
        Convert a source chunk into a retrieval result.

        Both the complete chunk and direct metadata fields are populated.
        This supports consumers that use either representation.
        """

        return RetrievalResult(
            chunk=chunk,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            expert_name=chunk.expert_name,
            expert_role=chunk.expert_role,
            market=chunk.market,
            text=chunk.text,
            start_timestamp=chunk.start_timestamp,
            end_timestamp=chunk.end_timestamp,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
            score=float(score),
            retrieval_score=float(score),
            retrieval_rank=rank,
            rank=rank,
        )

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        expected_count: int,
    ) -> None:
        """Validate embedding matrix."""

        if not isinstance(
            embeddings,
            np.ndarray,
        ):
            raise TypeError(
                "Embedding service must return a NumPy array."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Expected 2D embedding matrix, "
                f"got shape {embeddings.shape}."
            )

        if embeddings.shape[0] != expected_count:
            raise ValueError(
                "Embedding count does not match metadata count. "
                f"Expected {expected_count}, "
                f"got {embeddings.shape[0]}."
            )

        if embeddings.shape[1] <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        if not np.isfinite(
            embeddings
        ).all():
            raise ValueError(
                "Embedding matrix contains NaN or infinite values."
            )

    def _validate_loaded_metadata(
        self,
        metadata: dict[str, Any],
        index: faiss.Index,
    ) -> None:
        """Validate persisted metadata against the FAISS index."""

        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "Vector metadata must be a JSON object."
            )

        stored_model = metadata.get(
            "embedding_model"
        )

        if stored_model is not None:
            current_model = (
                self.embedding_service.model_name
            )

            if stored_model != current_model:
                raise ValueError(
                    "Embedding model mismatch. Persisted index uses "
                    f"'{stored_model}', while the application uses "
                    f"'{current_model}'."
                )

        stored_dimension = metadata.get(
            "embedding_dimension"
        )

        if (
            stored_dimension is not None
            and int(stored_dimension) != index.d
        ):
            raise ValueError(
                "Stored embedding dimension does not match "
                "FAISS index."
            )

        if index.ntotal <= 0:
            raise ValueError(
                "Persisted FAISS index is empty."
            )

    @staticmethod
    def _truncate_for_log(
        value: str,
        max_length: int = 120,
    ) -> str:
        """Keep long query strings manageable in logs."""

        normalized = " ".join(
            value.split()
        )

        if len(normalized) <= max_length:
            return normalized

        return (
            f"{normalized[:max_length - 3]}..."
        )


def _sys_module():
    """Return sys for compatibility with SensorException."""

    import sys

    return sys


__all__ = [
    "FAISSVectorStore",
]