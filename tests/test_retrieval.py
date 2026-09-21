"""
Unit tests for transcript retrieval.

These tests verify that the retrieval layer:

- returns relevant transcript chunks;
- preserves source metadata;
- supports expert filtering;
- supports market filtering;
- supports document filtering;
- handles empty/unknown queries safely;
- respects retrieval limits;
- returns deterministic metadata for indexed documents.

The tests use a small in-memory FAISS index so they do not depend on
the production vector-store files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import TranscriptChunk
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import RetrievalFilters, Retriever
from src.retrieval.vector_store import FAISSVectorStore


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """
    Create the embedding service used by retrieval tests.

    The production embedding model is intentionally reused so the tests
    exercise the same embedding interface used by the application.
    """
    return EmbeddingService()


@pytest.fixture
def sample_chunks() -> list[TranscriptChunk]:
    """Return representative transcript chunks from all three markets."""
    return [
        TranscriptChunk(
            chunk_id="france-001",
            document_id="france-doc",
            source_file="Transcript_1_France.txt",
            expert_name="Dr. Jean Martin",
            expert_role="Head of Urology",
            market="France",
            text=(
                "Adoption is growing, concentrated in larger academic and "
                "private centres with stronger capital budgets."
            ),
            start_timestamp="00:18",
            end_timestamp="01:20",
            start_seconds=18.0,
            end_seconds=80.0,
            segment_ids=["france-segment-001"],
        ),
        TranscriptChunk(
            chunk_id="france-002",
            document_id="france-doc",
            source_file="Transcript_1_France.txt",
            expert_name="Dr. Jean Martin",
            expert_role="Head of Urology",
            market="France",
            text=(
                "The biggest barrier is capital budget approval. "
                "Hospitals need a clear economic case."
            ),
            start_timestamp="01:20",
            end_timestamp="02:18",
            start_seconds=80.0,
            end_seconds=138.0,
            segment_ids=["france-segment-002"],
        ),
        TranscriptChunk(
            chunk_id="france-003",
            document_id="france-doc",
            source_file="Transcript_1_France.txt",
            expert_name="Dr. Jean Martin",
            expert_role="Head of Urology",
            market="France",
            text=(
                "ROI is very important. Finance wants utilisation, "
                "procedure volume, maintenance cost, and payback."
            ),
            start_timestamp="02:18",
            end_timestamp="03:10",
            start_seconds=138.0,
            end_seconds=190.0,
            segment_ids=["france-segment-003"],
        ),
        TranscriptChunk(
            chunk_id="germany-001",
            document_id="germany-doc",
            source_file="Transcript_2_Germany.txt",
            expert_name="Anna Keller",
            expert_role="Former Hospital Procurement Director",
            market="Germany",
            text=(
                "The market is growing but uneven. Large university hospitals "
                "are more advanced, while smaller hospitals are waiting."
            ),
            start_timestamp="00:16",
            end_timestamp="01:10",
            start_seconds=16.0,
            end_seconds=70.0,
            segment_ids=["germany-segment-001"],
        ),
        TranscriptChunk(
            chunk_id="germany-002",
            document_id="germany-doc",
            source_file="Transcript_2_Germany.txt",
            expert_name="Anna Keller",
            expert_role="Former Hospital Procurement Director",
            market="Germany",
            text=(
                "Cost is the first barrier. Capital purchases are difficult "
                "when hospital finances are under pressure."
            ),
            start_timestamp="01:10",
            end_timestamp="02:08",
            start_seconds=70.0,
            end_seconds=128.0,
            segment_ids=["germany-segment-002"],
        ),
        TranscriptChunk(
            chunk_id="germany-003",
            document_id="germany-doc",
            source_file="Transcript_2_Germany.txt",
            expert_name="Anna Keller",
            expert_role="Former Hospital Procurement Director",
            market="Germany",
            text=(
                "Total cost of ownership, expected procedure volume, "
                "maintenance, service contracts, and training are important."
            ),
            start_timestamp="02:08",
            end_timestamp="03:05",
            start_seconds=128.0,
            end_seconds=185.0,
            segment_ids=["germany-segment-003"],
        ),
        TranscriptChunk(
            chunk_id="uk-001",
            document_id="uk-doc",
            source_file="Transcript_3_UK.txt",
            expert_name="Dr. Emily Carter",
            expert_role="Consultant Urologist",
            market="UK",
            text=(
                "Robotic surgery is increasing. Some larger NHS trusts "
                "have made it standard for selected procedures."
            ),
            start_timestamp="00:14",
            end_timestamp="01:05",
            start_seconds=14.0,
            end_seconds=65.0,
            segment_ids=["uk-segment-001"],
        ),
        TranscriptChunk(
            chunk_id="uk-002",
            document_id="uk-doc",
            source_file="Transcript_3_UK.txt",
            expert_name="Dr. Emily Carter",
            expert_role="Consultant Urologist",
            market="UK",
            text=(
                "Funding is important, but training capacity is equally "
                "important. Without trained surgeons and theatre staff, "
                "adoption stalls."
            ),
            start_timestamp="01:05",
            end_timestamp="02:07",
            start_seconds=65.0,
            end_seconds=127.0,
            segment_ids=["uk-segment-002"],
        ),
        TranscriptChunk(
            chunk_id="uk-003",
            document_id="uk-doc",
            source_file="Transcript_3_UK.txt",
            expert_name="Dr. Emily Carter",
            expert_role="Consultant Urologist",
            market="UK",
            text=(
                "ROI matters, but it is not purely financial. Patient "
                "outcomes, length of stay, surgeon recruitment, and "
                "clinical position also matter."
            ),
            start_timestamp="02:07",
            end_timestamp="03:10",
            start_seconds=127.0,
            end_seconds=190.0,
            segment_ids=["uk-segment-003"],
        ),
    ]


@pytest.fixture
def vector_store(
    embedding_service: EmbeddingService,
    sample_chunks: list[TranscriptChunk],
    tmp_path: Path,
) -> FAISSVectorStore:
    """Build a temporary vector store for retrieval tests."""
    texts = [chunk.text for chunk in sample_chunks]

    embeddings = embedding_service.embed_documents(texts)

    store = FAISSVectorStore(
        storage_dir=tmp_path / "vector_store",
    )

    store.build(
        embeddings=embeddings,
        metadata=[
            chunk.model_dump(mode="json")
            for chunk in sample_chunks
        ],
    )

    return store


@pytest.fixture
def retriever(
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
) -> Retriever:
    """Create a Retriever backed by the temporary vector store."""
    return Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=8,
    )


def test_retrieval_returns_results(
    retriever: Retriever,
) -> None:
    """A relevant query should return at least one result."""
    results = retriever.retrieve(
        "What is the main barrier to adoption?"
    )

    assert results
    assert len(results) > 0


def test_retrieval_result_contains_source_metadata(
    retriever: Retriever,
) -> None:
    """Retrieved results must retain transcript provenance."""
    results = retriever.retrieve(
        "capital budget approval"
    )

    assert results

    result = results[0]

    assert result.chunk_id
    assert result.document_id
    assert result.source_file
    assert result.expert_name
    assert result.market
    assert result.text


def test_retrieval_finds_relevant_france_evidence(
    retriever: Retriever,
) -> None:
    """A France-specific query should retrieve France evidence."""
    results = retriever.retrieve(
        "France capital budget approval barrier"
    )

    assert results

    source_files = {
        result.source_file
        for result in results
    }

    assert "Transcript_1_France.txt" in source_files


def test_retrieval_finds_relevant_germany_evidence(
    retriever: Retriever,
) -> None:
    """A Germany-specific query should retrieve Germany evidence."""
    results = retriever.retrieve(
        "Germany hospital cost capital purchase"
    )

    assert results

    source_files = {
        result.source_file
        for result in results
    }

    assert "Transcript_2_Germany.txt" in source_files


def test_retrieval_finds_relevant_uk_evidence(
    retriever: Retriever,
) -> None:
    """A UK-specific query should retrieve UK evidence."""
    results = retriever.retrieve(
        "UK funding training capacity"
    )

    assert results

    source_files = {
        result.source_file
        for result in results
    }

    assert "Transcript_3_UK.txt" in source_files


def test_market_filter_limits_results(
    retriever: Retriever,
) -> None:
    """Market filtering should exclude other markets."""
    filters = RetrievalFilters(
        market="France",
    )

    results = retriever.retrieve(
        "adoption barriers",
        filters=filters,
    )

    assert results

    assert all(
        result.market == "France"
        for result in results
    )


def test_expert_filter_limits_results(
    retriever: Retriever,
) -> None:
    """Expert filtering should exclude other experts."""
    filters = RetrievalFilters(
        expert_name="Anna Keller",
    )

    results = retriever.retrieve(
        "cost and purchasing",
        filters=filters,
    )

    assert results

    assert all(
        result.expert_name == "Anna Keller"
        for result in results
    )


def test_document_filter_limits_results(
    retriever: Retriever,
) -> None:
    """Document filtering should return only the requested transcript."""
    filters = RetrievalFilters(
        document_id="uk-doc",
    )

    results = retriever.retrieve(
        "robotic surgery adoption",
        filters=filters,
    )

    assert results

    assert all(
        result.document_id == "uk-doc"
        for result in results
    )


def test_source_file_filter_limits_results(
    retriever: Retriever,
) -> None:
    """Source-file filtering should return only matching evidence."""
    filters = RetrievalFilters(
        source_file="Transcript_1_France.txt",
    )

    results = retriever.retrieve(
        "hospital economics",
        filters=filters,
    )

    assert results

    assert all(
        result.source_file == "Transcript_1_France.txt"
        for result in results
    )


def test_combined_filters_are_respected(
    retriever: Retriever,
) -> None:
    """Multiple retrieval filters should be applied together."""
    filters = RetrievalFilters(
        market="France",
        expert_name="Dr. Jean Martin",
        document_id="france-doc",
    )

    results = retriever.retrieve(
        "ROI economics",
        filters=filters,
    )

    assert results

    assert all(
        result.market == "France"
        and result.expert_name == "Dr. Jean Martin"
        and result.document_id == "france-doc"
        for result in results
    )


def test_retrieval_respects_top_k(
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
) -> None:
    """Retriever should not return more than configured top_k results."""
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=3,
    )

    results = retriever.retrieve(
        "robotic surgery adoption",
    )

    assert len(results) <= 3


def test_retrieval_scores_are_numeric(
    retriever: Retriever,
) -> None:
    """Every retrieval result should contain a numeric relevance score."""
    results = retriever.retrieve(
        "training and adoption",
    )

    assert results

    assert all(
        isinstance(result.score, float)
        for result in results
    )


def test_retrieval_results_are_ranked(
    retriever: Retriever,
) -> None:
    """Results should be returned in descending score order."""
    results = retriever.retrieve(
        "ROI procedure volume maintenance payback",
    )

    assert results

    scores = [result.score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_expert_retrieval_helper(
    retriever: Retriever,
) -> None:
    """The expert-specific helper should apply expert filtering."""
    results = retriever.retrieve_for_expert(
        query="training",
        expert_name="Dr. Emily Carter",
    )

    assert results

    assert all(
        result.expert_name == "Dr. Emily Carter"
        for result in results
    )


def test_market_retrieval_helper(
    retriever: Retriever,
) -> None:
    """The market-specific helper should apply market filtering."""
    results = retriever.retrieve_for_market(
        query="adoption",
        market="Germany",
    )

    assert results

    assert all(
        result.market == "Germany"
        for result in results
    )


def test_all_experts_retrieval_helper(
    retriever: Retriever,
) -> None:
    """Cross-expert retrieval should allow multiple expert sources."""
    results = retriever.retrieve_for_all_experts(
        query="ROI purchasing economics",
    )

    assert results

    experts = {
        result.expert_name
        for result in results
    }

    assert len(experts) >= 2


def test_unknown_market_returns_no_results(
    retriever: Retriever,
) -> None:
    """A filter for an unknown market should fail safely."""
    filters = RetrievalFilters(
        market="Unknown Market",
    )

    results = retriever.retrieve(
        "adoption",
        filters=filters,
    )

    assert results == []


def test_unknown_expert_returns_no_results(
    retriever: Retriever,
) -> None:
    """A filter for an unknown expert should fail safely."""
    filters = RetrievalFilters(
        expert_name="Unknown Expert",
    )

    results = retriever.retrieve(
        "adoption",
        filters=filters,
    )

    assert results == []


def test_retrieval_does_not_modify_source_metadata(
    retriever: Retriever,
) -> None:
    """Retrieval should preserve original source metadata."""
    results = retriever.retrieve(
        "capital budget approval",
    )

    assert results

    result = results[0]

    assert result.source_file.endswith(".txt")
    assert result.expert_name
    assert result.market


def test_retrieval_preserves_timestamp_metadata(
    retriever: Retriever,
) -> None:
    """Retrieved chunks should retain their timestamp information."""
    results = retriever.retrieve(
        "capital budget approval",
    )

    assert results

    result = results[0]

    assert result.start_timestamp
    assert result.start_seconds is not None


def test_retrieval_preserves_chunk_identity(
    retriever: Retriever,
) -> None:
    """Retrieved results should expose their original chunk IDs."""
    results = retriever.retrieve(
        "capital budget approval",
    )

    assert results

    assert all(
        result.chunk_id
        for result in results
    )


def test_vector_store_count_matches_indexed_chunks(
    vector_store: FAISSVectorStore,
    sample_chunks: list[TranscriptChunk],
) -> None:
    """The temporary vector store should contain every test chunk."""
    assert vector_store.count == len(sample_chunks)


def test_vector_store_dimension_matches_embeddings(
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
) -> None:
    """FAISS index dimension should match embedding dimension."""
    assert vector_store.dimension == embedding_service.dimension


def test_normalized_embedding_similarity_is_bounded(
    embedding_service: EmbeddingService,
) -> None:
    """Normalized embeddings should produce cosine-compatible vectors."""
    embeddings = embedding_service.embed_documents(
        [
            "capital budget approval",
            "hospital financial approval",
        ]
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    )


def test_retrieval_is_repeatable(
    retriever: Retriever,
) -> None:
    """Repeated retrieval should return stable metadata ordering."""
    query = "hospital purchasing timeline"

    first = retriever.retrieve(query)
    second = retriever.retrieve(query)

    assert [
        result.chunk_id
        for result in first
    ] == [
        result.chunk_id
        for result in second
    ]


def test_empty_query_fails_safely(
    retriever: Retriever,
) -> None:
    """An empty query should not silently retrieve arbitrary evidence."""
    try:
        results = retriever.retrieve("")
    except Exception:
        return

    assert results == []


def test_whitespace_only_query_fails_safely(
    retriever: Retriever,
) -> None:
    """Whitespace-only queries should not retrieve arbitrary evidence."""
    try:
        results = retriever.retrieve("   ")
    except Exception:
        return

    assert results == []


def test_cross_market_query_can_return_multiple_markets(
    retriever: Retriever,
) -> None:
    """
    A broad cross-market query should be capable of returning evidence
    from more than one market.
    """
    results = retriever.retrieve(
        "robotic surgery adoption and hospital economics",
    )

    assert results

    markets = {
        result.market
        for result in results
    }

    assert len(markets) >= 2