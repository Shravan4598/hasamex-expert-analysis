"""
Core Pydantic models for Hasamex Expert Analysis.

The models intentionally support both:
1. the production pipeline contracts, and
2. lightweight construction used by evaluation/test layers.

The goal is to keep provenance, timestamps, expert identity,
and evidence traceability available throughout the application.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ======================================================================
# ENUMS
# ======================================================================


class SpeakerType(str, Enum):
    """Classification of a transcript speaker."""

    EXPERT = "expert"
    INTERVIEWER = "interviewer"
    MODERATOR = "moderator"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    """Verification state of evidence."""

    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    INSUFFICIENT = "insufficient"


class DifferenceType(str, Enum):
    """Relationship between expert positions."""

    DIRECT_DISAGREEMENT = "direct_disagreement"
    DIFFERENT_EMPHASIS = "different_emphasis"
    DIFFERENT_EXPERIENCE = "different_experience"
    COMPLEMENTARY = "complementary"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ======================================================================
# TRANSCRIPT MODELS
# ======================================================================


class TranscriptMetadata(BaseModel):
    """Metadata describing a transcript."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    document_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)

    expert_name: str = Field(..., min_length=1)
    expert_role: Optional[str] = None
    market: str = Field(..., min_length=1)

    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
    )

    segment_count: int = Field(
        default=0,
        ge=0,
    )


class TranscriptSegment(BaseModel):
    """One timestamped transcript segment."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    segment_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)

    source_file: str = Field(..., min_length=1)

    expert_name: Optional[str] = None
    expert_role: Optional[str] = None
    market: Optional[str] = None

    speaker: str = Field(
        default="Expert",
        min_length=1,
    )

    speaker_type: SpeakerType = SpeakerType.UNKNOWN

    text: str = Field(..., min_length=1)

    start_timestamp: str = Field(..., min_length=1)
    end_timestamp: Optional[str] = None

    start_seconds: float = Field(
        ...,
        ge=0.0,
    )

    end_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Return segment duration when an end timestamp exists."""

        if self.end_seconds is None:
            return None

        return max(
            0.0,
            float(self.end_seconds)
            - float(self.start_seconds),
        )


class TranscriptChunk(BaseModel):
    """Retrieval-ready transcript chunk."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)

    expert_name: Optional[str] = None
    expert_role: Optional[str] = None
    market: Optional[str] = None

    text: str = Field(..., min_length=1)

    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    segment_ids: List[str] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


# ======================================================================
# RETRIEVAL
# ======================================================================


class RetrievalResult(BaseModel):
    """
    Retrieval result with complete source provenance.

    Supports both direct construction and construction from
    a TranscriptChunk.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    # Full chunk when available from the vector store.
    chunk: Optional[TranscriptChunk] = None

    # Direct provenance fields are also retained because retrieval
    # results may be serialized independently of their chunk object.
    chunk_id: Optional[str] = None

    document_id: Optional[str] = None
    source_file: Optional[str] = None

    expert_name: Optional[str] = None
    expert_role: Optional[str] = None
    market: Optional[str] = None

    text: Optional[str] = None

    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    # Retrieval ranking/provenance.
    score: float = 0.0

    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None

    retrieval_rank: Optional[int] = None
    rank: Optional[int] = None


# ======================================================================
# EVIDENCE
# ======================================================================


class Evidence(BaseModel):
    """Verified evidence extracted from a transcript."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    evidence_id: str = Field(..., min_length=1)

    document_id: str = Field(..., min_length=1)

    chunk_id: Optional[str] = None

    source_file: str = Field(..., min_length=1)

    expert_name: Optional[str] = None
    expert_role: Optional[str] = None
    market: Optional[str] = None

    quote: str = Field(..., min_length=1)

    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    status: EvidenceStatus = EvidenceStatus.UNVERIFIED

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class Citation(BaseModel):
    """Citation attached to a grounded answer."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    citation_id: Optional[str] = None

    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    source_file: Optional[str] = None

    expert_name: Optional[str] = None
    expert_role: Optional[str] = None
    market: Optional[str] = None

    quote: Optional[str] = None

    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    evidence_id: Optional[str] = None

    text: Optional[str] = None


# ======================================================================
# INTERVIEW GUIDE
# ======================================================================


class InterviewQuestion(BaseModel):
    """Interview-guide question."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    question_id: str = Field(..., min_length=1)

    question: str = Field(..., min_length=1)

    # Optional numeric ordering used by the interview guide.
    question_number: Optional[int] = None

    category: Optional[str] = None


# ======================================================================
# GROUNDED ANSWER
# ======================================================================


class GroundedAnswer(BaseModel):
    """Answer constrained by transcript evidence."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    question_id: Optional[str] = None

    question: str = Field(..., min_length=1)

    answer: str = Field(..., min_length=1)

    evidence: List[Evidence] = Field(
        default_factory=list
    )

    citations: List[Citation] = Field(
        default_factory=list
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    evidence_status: EvidenceStatus = (
        EvidenceStatus.INSUFFICIENT
    )

    insufficient_evidence: bool = False

    reasoning: Optional[str] = None


# ======================================================================
# THEME EVIDENCE
# ======================================================================


class ThemeEvidence(BaseModel):
    """
    Evidence supporting a cross-expert theme.

    This is intentionally distinct from Evidence because theme
    analysis may operate on a lighter source representation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    expert_name: str = Field(..., min_length=1)
    market: str = Field(..., min_length=1)

    source_file: str = Field(..., min_length=1)

    start_timestamp: str = Field(..., min_length=1)

    quote: str = Field(..., min_length=1)

    # Optional analytical fields.
    evidence: Optional[str] = None

    relevance: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ======================================================================
# THEMES
# ======================================================================


class Theme(BaseModel):
    """Common theme identified across interviews."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    theme_id: Optional[str] = None

    # Test/UI contract uses "name"; production code may use "theme".
    name: str = Field(..., min_length=1)

    summary: str = Field(..., min_length=1)

    experts: List[str] = Field(
        default_factory=list
    )

    evidence: List[ThemeEvidence] = Field(
        default_factory=list
    )

    @property
    def theme(self) -> str:
        """Backward-compatible alias for the theme name."""

        return self.name


# ======================================================================
# EXPERT POSITIONS
# ======================================================================


class ExpertPosition(BaseModel):
    """One expert's position on a topic."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    expert_name: str = Field(..., min_length=1)

    market: str = Field(..., min_length=1)

    position: str = Field(..., min_length=1)

    # ThemeEvidence is accepted here because cross-expert
    # disagreement analysis may use lightweight source evidence.
    evidence: List[
        Union[Evidence, ThemeEvidence]
    ] = Field(
        default_factory=list
    )


# ======================================================================
# DISAGREEMENTS
# ======================================================================


class Disagreement(BaseModel):
    """Difference between expert positions."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    disagreement_id: Optional[str] = None

    topic: str = Field(..., min_length=1)

    difference_type: DifferenceType

    description: Optional[str] = None

    summary: Optional[str] = None

    positions: List[ExpertPosition] = Field(
        default_factory=list
    )

    evidence: List[
        Union[Evidence, ThemeEvidence]
    ] = Field(
        default_factory=list
    )


# ======================================================================
# SEARCH QUERY
# ======================================================================


class SearchQuery(BaseModel):
    """Cross-transcript search query."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    query: str = Field(..., min_length=1)

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    market: Optional[str] = None

    expert_name: Optional[str] = None

    # Backward-compatible filter names.
    market_filter: Optional[str] = None
    expert_filter: Optional[str] = None


# ======================================================================
# EVALUATION CASE
# ======================================================================


class EvaluationCase(BaseModel):
    """One reproducible evaluation case."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    case_id: str = Field(..., min_length=1)

    case_type: str = "qa"

    question: str = Field(..., min_length=1)

    expert_name: Optional[str] = None
    market: Optional[str] = None

    expected_expert: Optional[str] = None
    expected_market: Optional[str] = None

    expected_timestamp: Optional[str] = None

    expected_keywords: List[str] = Field(
        default_factory=list
    )

    expected_quote_fragments: List[str] = Field(
        default_factory=list
    )

    # Compatibility with implementations that use one complete quote.
    expected_quote: Optional[str] = None


# ======================================================================
# EVALUATION RESULT
# ======================================================================


class EvaluationResult(BaseModel):
    """Evaluation result for one test case."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    case_id: str = Field(..., min_length=1)

    # Execution-level result.
    passed: bool = False

    execution_success: bool = False

    # Evidence/citation checks.
    evidence_present: bool = False
    citations_present: bool = False
    evidence_sufficient: bool = False

    expert_match: bool = False
    market_match: bool = False

    timestamp_match: bool = False
    keyword_match: bool = False
    quote_match: bool = False

    evidence_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    failure_reasons: List[str] = Field(
        default_factory=list
    )

    # Detailed evaluation metrics.
    retrieval_relevance: float = 0.0

    citation_correctness: float = 0.0

    timestamp_correctness: float = 0.0

    quote_correctness: float = 0.0

    groundedness: float = 0.0

    completeness: float = 0.0


# ======================================================================
# APPLICATION STATISTICS
# ======================================================================


class ApplicationStats(BaseModel):
    """Runtime statistics for the indexed transcript corpus."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    transcript_count: int = Field(
        default=0,
        ge=0,
    )

    segment_count: int = Field(
        default=0,
        ge=0,
    )

    chunk_count: int = Field(
        default=0,
        ge=0,
    )

    expert_count: int = Field(
        default=0,
        ge=0,
    )

    market_count: int = Field(
        default=0,
        ge=0,
    )