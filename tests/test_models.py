"""
Unit tests for application data models.

These tests verify the Pydantic contracts used throughout the
Hasamex Expert Analysis application.

The model layer is important because it provides a consistent
contract between:

    ingestion -> retrieval -> evidence -> analysis -> UI -> evaluation

The tests focus on:

- valid model construction;
- required fields;
- enum values;
- validation behavior;
- serialization/deserialization;
- nested evidence;
- grounded answers;
- evaluation models;
- application statistics.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    ApplicationStats,
    Citation,
    DifferenceType,
    Disagreement,
    EvaluationCase,
    EvaluationResult,
    Evidence,
    EvidenceStatus,
    ExpertPosition,
    GroundedAnswer,
    InterviewQuestion,
    RetrievalResult,
    SearchQuery,
    SpeakerType,
    Theme,
    ThemeEvidence,
    TranscriptChunk,
    TranscriptMetadata,
    TranscriptSegment,
)

# ---------------------------------------------------------------------------
# TranscriptMetadata
# ---------------------------------------------------------------------------


def test_transcript_metadata_can_be_created() -> None:
    """Transcript metadata should accept a valid transcript description."""
    metadata = TranscriptMetadata(
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        duration_seconds=368.0,
    )

    assert metadata.document_id == "france-doc"
    assert metadata.source_file == "Transcript_1_France.txt"
    assert metadata.expert_name == "Dr. Jean Martin"
    assert metadata.market == "France"
    assert metadata.duration_seconds == 368.0


def test_transcript_metadata_supports_serialization() -> None:
    """Transcript metadata should round-trip through JSON-compatible data."""
    metadata = TranscriptMetadata(
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        duration_seconds=368.0,
    )

    payload = metadata.model_dump(mode="json")
    restored = TranscriptMetadata.model_validate(payload)

    assert restored == metadata


def test_transcript_metadata_requires_source_file() -> None:
    """Source-file attribution is required for traceability."""
    with pytest.raises(ValidationError):
        TranscriptMetadata(
            document_id="france-doc",
            source_file="",
            expert_name="Dr. Jean Martin",
            expert_role="Head of Urology",
            market="France",
        )


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


def test_transcript_segment_can_be_created() -> None:
    """A valid transcript segment should satisfy the model contract."""
    segment = TranscriptSegment(
        segment_id="france-segment-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        speaker="Expert",
        speaker_type=SpeakerType.EXPERT,
        text="Adoption is growing.",
        start_timestamp="00:18",
        end_timestamp="01:20",
        start_seconds=18.0,
        end_seconds=80.0,
    )

    assert segment.segment_id == "france-segment-001"
    assert segment.speaker_type == SpeakerType.EXPERT
    assert segment.start_seconds == 18.0
    assert segment.end_seconds == 80.0


def test_transcript_segment_accepts_enum_value() -> None:
    """Pydantic should accept the serialized enum value."""
    segment = TranscriptSegment(
        segment_id="segment-001",
        document_id="document-001",
        source_file="transcript.txt",
        expert_name="Expert",
        expert_role="Consultant",
        market="France",
        speaker="Expert",
        speaker_type="expert",
        text="A statement.",
        start_timestamp="00:18",
        start_seconds=18.0,
    )

    assert segment.speaker_type == SpeakerType.EXPERT


def test_transcript_segment_allows_missing_end_timestamp() -> None:
    """The final segment may not have a known end timestamp."""
    segment = TranscriptSegment(
        segment_id="segment-001",
        document_id="document-001",
        source_file="transcript.txt",
        expert_name="Expert",
        expert_role="Consultant",
        market="France",
        speaker="Expert",
        speaker_type=SpeakerType.EXPERT,
        text="Final statement.",
        start_timestamp="06:08",
        start_seconds=368.0,
    )

    assert segment.end_timestamp is None
    assert segment.end_seconds is None


def test_transcript_segment_requires_text() -> None:
    """A transcript segment cannot exist without source text."""
    with pytest.raises(ValidationError):
        TranscriptSegment(
            segment_id="segment-001",
            document_id="document-001",
            source_file="transcript.txt",
            expert_name="Expert",
            expert_role="Consultant",
            market="France",
            speaker="Expert",
            speaker_type=SpeakerType.EXPERT,
            text="",
            start_timestamp="00:18",
            start_seconds=18.0,
        )


# ---------------------------------------------------------------------------
# TranscriptChunk
# ---------------------------------------------------------------------------


def test_transcript_chunk_can_be_created() -> None:
    """A valid chunk should retain transcript provenance."""
    chunk = TranscriptChunk(
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        text="The biggest barrier is capital budget approval.",
        start_timestamp="01:20",
        end_timestamp="02:18",
        start_seconds=80.0,
        end_seconds=138.0,
        segment_ids=["segment-001"],
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.segment_ids == ["segment-001"]
    assert chunk.market == "France"


def test_transcript_chunk_round_trips() -> None:
    """Chunks should serialize and deserialize without losing metadata."""
    chunk = TranscriptChunk(
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        text="ROI is very important.",
        start_timestamp="02:18",
        end_timestamp="03:10",
        start_seconds=138.0,
        end_seconds=190.0,
        segment_ids=["segment-003"],
    )

    restored = TranscriptChunk.model_validate(
        chunk.model_dump(mode="json")
    )

    assert restored == chunk


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


def test_retrieval_result_can_be_created() -> None:
    """A retrieval result should expose both relevance and provenance."""
    result = RetrievalResult(
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        text="Capital budget approval is the biggest barrier.",
        start_timestamp="01:20",
        end_timestamp="02:18",
        start_seconds=80.0,
        end_seconds=138.0,
        score=0.91,
    )

    assert result.chunk_id == "chunk-001"
    assert result.score == pytest.approx(0.91)
    assert result.market == "France"


def test_retrieval_result_score_is_numeric() -> None:
    """Retrieval scores should be numeric."""
    result = RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc",
        source_file="transcript.txt",
        expert_name="Expert",
        expert_role="Role",
        market="France",
        text="Some evidence.",
        start_timestamp="00:18",
        start_seconds=18.0,
        score=0.75,
    )

    assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_can_be_created() -> None:
    """Verified evidence should contain source and quote metadata."""
    evidence = Evidence(
        evidence_id="evidence-001",
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="01:20",
        end_timestamp="02:18",
        start_seconds=80.0,
        end_seconds=138.0,
        quote="The biggest barrier is capital budget approval.",
        source_text="The biggest barrier is capital budget approval.",
        verification_status=EvidenceStatus.VERIFIED,
        verification_message="Exact quote verified.",
    )

    assert evidence.evidence_id == "evidence-001"
    assert evidence.verification_status == EvidenceStatus.VERIFIED
    assert evidence.quote


def test_evidence_can_be_unverified() -> None:
    """Evidence may explicitly represent failed verification."""
    evidence = Evidence(
        evidence_id="evidence-002",
        chunk_id="chunk-002",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="01:20",
        start_seconds=80.0,
        quote="Fabricated statement.",
        source_text="Original statement.",
        verification_status=EvidenceStatus.UNVERIFIED,
        verification_message="Quote could not be verified.",
    )

    assert evidence.verification_status == EvidenceStatus.UNVERIFIED


def test_evidence_requires_source_file() -> None:
    """Evidence without source attribution should fail validation."""
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-001",
            chunk_id="chunk-001",
            document_id="france-doc",
            source_file="",
            expert_name="Dr. Jean Martin",
            expert_role="Head of Urology",
            market="France",
            start_timestamp="01:20",
            start_seconds=80.0,
            quote="A quote.",
            source_text="A quote.",
            verification_status=EvidenceStatus.VERIFIED,
        )


def test_evidence_round_trips() -> None:
    """Evidence should survive serialization/deserialization."""
    evidence = Evidence(
        evidence_id="evidence-001",
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="02:18",
        end_timestamp="03:10",
        start_seconds=138.0,
        end_seconds=190.0,
        quote="ROI is very important.",
        source_text="ROI is very important.",
        verification_status=EvidenceStatus.VERIFIED,
    )

    restored = Evidence.model_validate(
        evidence.model_dump(mode="json")
    )

    assert restored == evidence


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def test_citation_can_be_created() -> None:
    """Citations should identify the source and timestamp."""
    citation = Citation(
        citation_id="cite-001",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        market="France",
        start_timestamp="01:20",
        end_timestamp="02:18",
        quote="The biggest barrier is capital budget approval.",
    )

    assert citation.citation_id == "cite-001"
    assert citation.source_file == "Transcript_1_France.txt"
    assert citation.start_timestamp == "01:20"


def test_citation_round_trips() -> None:
    """Citation serialization should preserve all source metadata."""
    citation = Citation(
        citation_id="cite-001",
        source_file="Transcript_2_Germany.txt",
        expert_name="Anna Keller",
        market="Germany",
        start_timestamp="02:08",
        end_timestamp="03:05",
        quote="Total cost of ownership is important.",
    )

    restored = Citation.model_validate(
        citation.model_dump(mode="json")
    )

    assert restored == citation


# ---------------------------------------------------------------------------
# InterviewQuestion
# ---------------------------------------------------------------------------


def test_interview_question_can_be_created() -> None:
    """Interview-guide questions should have stable identifiers."""
    question = InterviewQuestion(
        question_id="q1",
        question="How would you describe current adoption of robotic surgery in your market?",
    )

    assert question.question_id == "q1"
    assert "current adoption" in question.question


def test_interview_question_requires_question_text() -> None:
    """An empty interview question should not be valid."""
    with pytest.raises(ValidationError):
        InterviewQuestion(
            question_id="q1",
            question="",
        )


# ---------------------------------------------------------------------------
# GroundedAnswer
# ---------------------------------------------------------------------------


def test_grounded_answer_can_be_created() -> None:
    """A grounded answer should contain evidence and citation metadata."""
    evidence = Evidence(
        evidence_id="evidence-001",
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="01:20",
        start_seconds=80.0,
        quote="The biggest barrier is capital budget approval.",
        source_text="The biggest barrier is capital budget approval.",
        verification_status=EvidenceStatus.VERIFIED,
    )

    citation = Citation(
        citation_id="cite-001",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        market="France",
        start_timestamp="01:20",
        quote="The biggest barrier is capital budget approval.",
    )

    answer = GroundedAnswer(
        question="What are the main barriers to adoption in France?",
        expert_name="Dr. Jean Martin",
        market="France",
        answer=(
            "The main barrier described is capital budget approval, "
            "with a clear economic case required."
        ),
        evidence=[evidence],
        citations=[citation],
        confidence=0.92,
        evidence_coverage=1.0,
        evidence_sufficient=True,
    )

    assert answer.expert_name == "Dr. Jean Martin"
    assert answer.market == "France"
    assert len(answer.evidence) == 1
    assert len(answer.citations) == 1
    assert answer.evidence_sufficient is True


def test_grounded_answer_can_represent_insufficient_evidence() -> None:
    """The model should support explicit fail-closed answers."""
    answer = GroundedAnswer(
        question="What exact market share percentage does France have?",
        answer="The supplied transcripts do not provide this information.",
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=(
            "No transcript evidence supports an exact market-share percentage."
        ),
    )

    assert answer.evidence_sufficient is False
    assert answer.refusal_reason
    assert answer.evidence == []
    assert answer.citations == []


def test_grounded_answer_round_trips() -> None:
    """Grounded answers should serialize nested evidence correctly."""
    answer = GroundedAnswer(
        question="What is the purchasing timeline?",
        expert_name="Anna Keller",
        market="Germany",
        answer="Nine to eighteen months is common.",
        evidence=[],
        citations=[],
        confidence=0.88,
        evidence_coverage=0.9,
        evidence_sufficient=True,
    )

    restored = GroundedAnswer.model_validate(
        answer.model_dump(mode="json")
    )

    assert restored == answer


# ---------------------------------------------------------------------------
# SearchQuery
# ---------------------------------------------------------------------------


def test_search_query_can_be_created() -> None:
    """A search query should represent a user request and optional filters."""
    query = SearchQuery(
        query="How important is ROI?",
        top_k=5,
        market="France",
        expert_name="Dr. Jean Martin",
    )

    assert query.query == "How important is ROI?"
    assert query.top_k == 5
    assert query.market == "France"
    assert query.expert_name == "Dr. Jean Martin"


def test_search_query_supports_cross_expert_search() -> None:
    """A search query may omit expert and market filters."""
    query = SearchQuery(
        query="What are the common adoption barriers?",
        top_k=8,
    )

    assert query.expert_name is None
    assert query.market is None


# ---------------------------------------------------------------------------
# ThemeEvidence
# ---------------------------------------------------------------------------


def test_theme_evidence_can_be_created() -> None:
    """Theme evidence should preserve expert and source context."""
    evidence = ThemeEvidence(
        expert_name="Dr. Jean Martin",
        market="France",
        source_file="Transcript_1_France.txt",
        start_timestamp="02:18",
        quote="ROI is very important.",
    )

    assert evidence.expert_name == "Dr. Jean Martin"
    assert evidence.market == "France"
    assert evidence.start_timestamp == "02:18"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def test_theme_can_be_created() -> None:
    """A theme should contain supporting evidence."""
    theme = Theme(
        theme_id="theme-001",
        name="Hospital Economics",
        summary="Hospital economics strongly influence purchasing decisions.",
        experts=["Dr. Jean Martin", "Anna Keller"],
        evidence=[
            ThemeEvidence(
                expert_name="Dr. Jean Martin",
                market="France",
                source_file="Transcript_1_France.txt",
                start_timestamp="02:18",
                quote="ROI is very important.",
            ),
            ThemeEvidence(
                expert_name="Anna Keller",
                market="Germany",
                source_file="Transcript_2_Germany.txt",
                start_timestamp="02:08",
                quote="Total cost of ownership is important.",
            ),
        ],
    )

    assert theme.theme_id == "theme-001"
    assert len(theme.experts) == 2
    assert len(theme.evidence) == 2


# ---------------------------------------------------------------------------
# ExpertPosition
# ---------------------------------------------------------------------------


def test_expert_position_can_be_created() -> None:
    """Expert positions should retain market-specific source context."""
    position = ExpertPosition(
        expert_name="Dr. Emily Carter",
        market="UK",
        position=(
            "Funding and training capacity are both important "
            "for sustainable adoption."
        ),
        evidence=[
            ThemeEvidence(
                expert_name="Dr. Emily Carter",
                market="UK",
                source_file="Transcript_3_UK.txt",
                start_timestamp="01:05",
                quote="Training capacity is equally important.",
            )
        ],
    )

    assert position.expert_name == "Dr. Emily Carter"
    assert position.market == "UK"
    assert len(position.evidence) == 1


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


def test_disagreement_can_be_created() -> None:
    """A disagreement model should classify the relationship explicitly."""
    disagreement = Disagreement(
        topic="Expected procedure-volume growth",
        difference_type=DifferenceType.DIRECT_DISAGREEMENT,
        summary="The experts describe different growth expectations.",
        positions=[],
        evidence=[],
    )

    assert disagreement.topic == "Expected procedure-volume growth"
    assert (
        disagreement.difference_type
        == DifferenceType.DIRECT_DISAGREEMENT
    )


def test_all_difference_types_are_available() -> None:
    """The model should expose all supported cross-expert classifications."""
    expected = {
        "direct_disagreement",
        "different_emphasis",
        "different_experience",
        "complementary",
        "insufficient_evidence",
    }

    actual = {
        difference.value
        for difference in DifferenceType
    }

    assert expected.issubset(actual)


# ---------------------------------------------------------------------------
# EvaluationCase
# ---------------------------------------------------------------------------


def test_evaluation_case_can_be_created() -> None:
    """An evaluation case should represent a reproducible test."""
    case = EvaluationCase(
        case_id="qa-france-001",
        case_type="qa",
        question="What are the main barriers to adoption in France?",
        expert_name="Dr. Jean Martin",
        market="France",
        expected_expert="Dr. Jean Martin",
        expected_market="France",
        expected_timestamp="01:20",
        expected_keywords=[
            "capital budget",
            "economic case",
        ],
        expected_quote_fragments=[
            "capital budget approval",
        ],
    )

    assert case.case_id == "qa-france-001"
    assert case.case_type == "qa"
    assert case.market == "France"
    assert "capital budget" in case.expected_keywords


def test_insufficient_evidence_case_can_have_no_expectations() -> None:
    """Unsupported-information tests may intentionally omit expected evidence."""
    case = EvaluationCase(
        case_id="unsupported-001",
        case_type="qa",
        question=(
            "What exact robotic surgery market share percentage "
            "does each country currently have?"
        ),
    )

    assert case.expected_expert is None
    assert case.expected_market is None
    assert case.expected_timestamp is None


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


def test_evaluation_result_can_be_created() -> None:
    """An evaluation result should summarize individual case execution."""
    result = EvaluationResult(
        case_id="qa-france-001",
        passed=True,
        execution_success=True,
        evidence_present=True,
        citations_present=True,
        evidence_sufficient=True,
        expert_match=True,
        market_match=True,
        timestamp_match=True,
        keyword_match=True,
        quote_match=True,
        evidence_coverage=1.0,
        failure_reasons=[],
    )

    assert result.case_id == "qa-france-001"
    assert result.passed is True
    assert result.execution_success is True


def test_failed_evaluation_result_can_store_reasons() -> None:
    """Failed evaluation cases should explain why they failed."""
    result = EvaluationResult(
        case_id="qa-france-002",
        passed=False,
        execution_success=True,
        evidence_present=False,
        citations_present=False,
        evidence_sufficient=False,
        expert_match=False,
        market_match=False,
        timestamp_match=False,
        keyword_match=False,
        quote_match=False,
        evidence_coverage=0.0,
        failure_reasons=[
            "No evidence was returned.",
            "No citations were returned.",
        ],
    )

    assert result.passed is False
    assert len(result.failure_reasons) == 2


# ---------------------------------------------------------------------------
# ApplicationStats
# ---------------------------------------------------------------------------


def test_application_stats_can_be_created() -> None:
    """Application statistics should represent actual corpus state."""
    stats = ApplicationStats(
        transcript_count=3,
        segment_count=21,
        chunk_count=15,
        expert_count=3,
        market_count=3,
    )

    assert stats.transcript_count == 3
    assert stats.segment_count == 21
    assert stats.chunk_count == 15
    assert stats.expert_count == 3
    assert stats.market_count == 3


def test_application_stats_support_zero_counts() -> None:
    """Empty-state statistics should remain representable."""
    stats = ApplicationStats(
        transcript_count=0,
        segment_count=0,
        chunk_count=0,
        expert_count=0,
        market_count=0,
    )

    assert stats.transcript_count == 0
    assert stats.segment_count == 0
    assert stats.chunk_count == 0


# ---------------------------------------------------------------------------
# General serialization tests
# ---------------------------------------------------------------------------


def test_evidence_status_values_are_serializable() -> None:
    """Evidence status enums should serialize to JSON-compatible values."""
    for status in EvidenceStatus:
        assert isinstance(status.value, str)


def test_speaker_type_values_are_serializable() -> None:
    """Speaker type enums should serialize to JSON-compatible values."""
    for speaker_type in SpeakerType:
        assert isinstance(speaker_type.value, str)


def test_difference_type_values_are_serializable() -> None:
    """Difference classification values should be JSON-compatible."""
    for difference_type in DifferenceType:
        assert isinstance(difference_type.value, str)


def test_nested_grounded_answer_serializes_to_json() -> None:
    """Nested answer models should produce JSON-compatible dictionaries."""
    evidence = Evidence(
        evidence_id="evidence-001",
        chunk_id="chunk-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="01:20",
        start_seconds=80.0,
        quote="The biggest barrier is capital budget approval.",
        source_text="The biggest barrier is capital budget approval.",
        verification_status=EvidenceStatus.VERIFIED,
    )

    answer = GroundedAnswer(
        question="What is the main barrier?",
        expert_name="Dr. Jean Martin",
        market="France",
        answer="Capital budget approval.",
        evidence=[evidence],
        citations=[],
        confidence=0.9,
        evidence_coverage=1.0,
        evidence_sufficient=True,
    )

    payload = answer.model_dump(mode="json")

    assert isinstance(payload, dict)
    assert isinstance(payload["evidence"], list)
    assert payload["evidence"][0]["verification_status"] == "verified"