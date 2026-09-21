"""
Tests for insufficient-evidence and fail-closed behavior.

The Hasamex Expert Analysis application must never fabricate information
that is not supported by the supplied expert transcripts.

These tests verify that:

- an answer can explicitly represent insufficient evidence;
- unsupported questions do not accidentally become grounded answers;
- empty evidence produces zero evidence coverage;
- unsupported answers do not contain citations;
- refusal reasons are preserved;
- fabricated evidence is not treated as verified;
- quote verification rejects source text that is not present;
- supported evidence remains distinguishable from unsupported evidence.

These tests intentionally avoid calling a real Gemini API or building a
real FAISS index. The behavior under test is the evidence/grounding
contract represented by the application's models and quote verifier.
"""

from __future__ import annotations

import pytest

from src.evidence.quote_verifier import QuoteVerifier
from src.models import (
    Citation,
    Evidence,
    EvidenceStatus,
    GroundedAnswer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unsupported_question() -> str:
    """Return a question whose answer is not present in the case pack."""
    return (
        "What exact robotic surgery market share percentage "
        "does each country currently have?"
    )


@pytest.fixture
def refusal_reason() -> str:
    """Return the expected style of fail-closed explanation."""
    return (
        "The supplied transcripts do not provide "
        "an exact robotic surgery market-share percentage."
    )


@pytest.fixture
def verified_evidence() -> Evidence:
    """Return a genuine evidence object from the supplied transcript."""
    return Evidence(
        evidence_id="evidence-france-001",
        chunk_id="chunk-france-001",
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


# ---------------------------------------------------------------------------
# GroundedAnswer: insufficient evidence
# ---------------------------------------------------------------------------


def test_insufficient_evidence_answer_has_no_evidence(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """Unsupported questions should be representable with zero evidence."""
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    assert answer.evidence == []
    assert answer.evidence_coverage == 0.0
    assert answer.evidence_sufficient is False


def test_insufficient_evidence_answer_has_no_citations(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """An unsupported answer must not manufacture source citations."""
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    assert answer.citations == []


def test_insufficient_evidence_answer_preserves_refusal_reason(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """The reason for refusing an unsupported question must be retained."""
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    assert answer.refusal_reason == refusal_reason


def test_insufficient_evidence_answer_has_zero_confidence(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """
    A fail-closed answer should not report confidence in an unsupported fact.
    """
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    assert answer.confidence == 0.0


def test_insufficient_evidence_round_trips_through_pydantic(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """Fail-closed answers should survive JSON serialization."""
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    payload = answer.model_dump(mode="json")
    restored = GroundedAnswer.model_validate(payload)

    assert restored == answer
    assert restored.evidence_sufficient is False
    assert restored.evidence == []
    assert restored.citations == []


# ---------------------------------------------------------------------------
# Evidence status safety
# ---------------------------------------------------------------------------


def test_fabricated_evidence_is_not_marked_verified() -> None:
    """Fabricated source text must be represented as unverified evidence."""
    evidence = Evidence(
        evidence_id="fabricated-001",
        chunk_id="chunk-france-001",
        document_id="france-doc",
        source_file="Transcript_1_France.txt",
        expert_name="Dr. Jean Martin",
        expert_role="Head of Urology",
        market="France",
        start_timestamp="01:20",
        start_seconds=80.0,
        quote="France has exactly 42% robotic surgery market share.",
        source_text=(
            "The biggest barrier is capital budget approval."
        ),
        verification_status=EvidenceStatus.UNVERIFIED,
        verification_message=(
            "Quote could not be verified against the source."
        ),
    )

    assert evidence.verification_status == EvidenceStatus.UNVERIFIED
    assert (
        "42%"
        in evidence.quote
    )
    assert (
        "42%"
        not in evidence.source_text
    )


def test_verified_evidence_requires_matching_source_text(
    verified_evidence: Evidence,
) -> None:
    """
    A verified evidence object should contain the same factual quote
    represented in its source text.
    """
    assert verified_evidence.verification_status == EvidenceStatus.VERIFIED
    assert verified_evidence.quote in verified_evidence.source_text


# ---------------------------------------------------------------------------
# QuoteVerifier: unsupported/fabricated claims
# ---------------------------------------------------------------------------


def test_quote_verifier_rejects_fabricated_quote() -> None:
    """A quote absent from the transcript must not verify."""
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote=(
            "France has exactly 42% robotic surgery market share."
        ),
        source_text=(
            "The biggest barrier is capital budget approval."
        ),
    )

    assert result.is_verified is False


def test_quote_verifier_accepts_exact_source_quote() -> None:
    """An exact transcript quote should verify successfully."""
    verifier = QuoteVerifier()

    source = (
        "The biggest barrier is capital budget approval."
    )

    result = verifier.verify_quote(
        quote=source,
        source_text=source,
    )

    assert result.is_verified is True


def test_quote_verifier_rejects_semantically_similar_but_absent_text() -> None:
    """
    Semantic similarity alone must not make an invented quote exact.

    The application requires an actual source-text match for exact quotes.
    """
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote=(
            "Capital budget approval is the primary obstacle "
            "to robotic surgery adoption."
        ),
        source_text=(
            "The biggest barrier is capital budget approval."
        ),
    )

    assert result.is_verified is False


def test_quote_verifier_rejects_fabricated_percentage() -> None:
    """A percentage not present in the transcript must be rejected."""
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote="Procedure volume increased by exactly 25%.",
        source_text=(
            "Procedure volume is increasing steadily."
        ),
    )

    assert result.is_verified is False


def test_quote_verifier_rejects_empty_source() -> None:
    """A quote cannot be verified against an empty source."""
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote="Any statement.",
        source_text="",
    )

    assert result.is_verified is False


def test_quote_verifier_rejects_empty_quote() -> None:
    """An empty quote cannot be considered verified evidence."""
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote="",
        source_text="The biggest barrier is capital budget approval.",
    )

    assert result.is_verified is False


# ---------------------------------------------------------------------------
# Citation safety
# ---------------------------------------------------------------------------


def test_unsupported_answer_does_not_create_fake_citation(
    unsupported_question: str,
    refusal_reason: str,
) -> None:
    """
    A refusal should contain no citation pretending to support
    an unavailable market-share percentage.
    """
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide this information."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=refusal_reason,
    )

    assert len(answer.citations) == 0


def test_supported_answer_can_contain_real_citation(
    verified_evidence: Evidence,
) -> None:
    """
    Supported information may have a citation when genuine evidence exists.
    """
    citation = Citation(
        citation_id="citation-france-001",
        source_file=verified_evidence.source_file,
        expert_name=verified_evidence.expert_name,
        market=verified_evidence.market,
        start_timestamp=verified_evidence.start_timestamp,
        end_timestamp=verified_evidence.end_timestamp,
        quote=verified_evidence.quote,
    )

    answer = GroundedAnswer(
        question="What is the main barrier to adoption in France?",
        expert_name=verified_evidence.expert_name,
        market=verified_evidence.market,
        answer=(
            "The main barrier described by the expert is "
            "capital budget approval."
        ),
        evidence=[verified_evidence],
        citations=[citation],
        confidence=0.95,
        evidence_coverage=1.0,
        evidence_sufficient=True,
    )

    assert answer.evidence_sufficient is True
    assert len(answer.evidence) == 1
    assert len(answer.citations) == 1
    assert answer.citations[0].source_file == (
        "Transcript_1_France.txt"
    )


# ---------------------------------------------------------------------------
# Grounding invariants
# ---------------------------------------------------------------------------


def test_insufficient_evidence_cannot_have_positive_evidence_coverage(
    unsupported_question: str,
) -> None:
    """
    An unsupported answer should not claim complete evidence coverage.
    """
    answer = GroundedAnswer(
        question=unsupported_question,
        answer="The transcripts do not provide this information.",
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=(
            "The requested information is absent from the transcripts."
        ),
    )

    assert answer.evidence_sufficient is False
    assert answer.evidence_coverage == 0.0


def test_supported_evidence_and_unsupported_answer_are_distinguishable(
    verified_evidence: Evidence,
) -> None:
    """The model should clearly distinguish grounded and refused answers."""
    supported = GroundedAnswer(
        question="What is the main barrier?",
        expert_name="Dr. Jean Martin",
        market="France",
        answer="Capital budget approval.",
        evidence=[verified_evidence],
        citations=[],
        confidence=0.9,
        evidence_coverage=1.0,
        evidence_sufficient=True,
    )

    unsupported = GroundedAnswer(
        question=(
            "What exact robotic surgery market share "
            "does France have?"
        ),
        answer="The transcripts do not provide this information.",
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=(
            "No exact market-share percentage is present "
            "in the supplied transcripts."
        ),
    )

    assert supported.evidence_sufficient is True
    assert supported.evidence
    assert unsupported.evidence_sufficient is False
    assert unsupported.evidence == []


def test_refusal_reason_should_not_be_empty_for_failed_grounding() -> None:
    """
    A failed-grounding answer should explain why the system could not answer.
    """
    answer = GroundedAnswer(
        question="What exact market share does France have?",
        answer="The transcripts do not provide this information.",
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=(
            "No exact market-share percentage is supported "
            "by the available transcript evidence."
        ),
    )

    assert answer.refusal_reason is not None
    assert answer.refusal_reason.strip()


# ---------------------------------------------------------------------------
# Regression tests for the actual case-pack limitation
# ---------------------------------------------------------------------------


def test_market_share_question_is_explicitly_unsupported(
    unsupported_question: str,
) -> None:
    """
    The case pack does not provide exact country-level market-share data.

    This test documents the intended behavior for the evaluation case:
    the application must refuse rather than invent a percentage.
    """
    answer = GroundedAnswer(
        question=unsupported_question,
        answer=(
            "The supplied transcripts do not provide exact "
            "robotic surgery market-share percentages for France, "
            "Germany, or the UK."
        ),
        evidence=[],
        citations=[],
        confidence=0.0,
        evidence_coverage=0.0,
        evidence_sufficient=False,
        refusal_reason=(
            "The transcripts discuss adoption trends but do not "
            "provide exact market-share percentages."
        ),
    )

    assert answer.evidence_sufficient is False
    assert answer.evidence_coverage == 0.0
    assert answer.confidence == 0.0
    assert answer.evidence == []
    assert answer.citations == []
    assert "market-share" in answer.refusal_reason.lower()


# ---------------------------------------------------------------------------
# Safety regression tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fabricated_quote",
    [
        "France has 42% market share.",
        "Germany has 55% market share.",
        "The UK has exactly 61% adoption.",
        "Robotic surgery adoption is exactly 73%.",
        "All hospitals use robotic surgery.",
    ],
)
def test_common_fabricated_claims_are_rejected(
    fabricated_quote: str,
) -> None:
    """
    Common unsupported quantitative/generalized claims must not verify
    against unrelated transcript evidence.
    """
    verifier = QuoteVerifier()

    result = verifier.verify_quote(
        quote=fabricated_quote,
        source_text=(
            "Adoption is growing, concentrated in larger "
            "academic and private centres."
        ),
    )

    assert result.is_verified is False