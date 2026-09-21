"""
Unit tests for quote verification.

These tests verify that exact quotes displayed by the application are
actually present in the original transcript evidence.

The quote verifier is an important hallucination-prevention component.
It should:

- accept exact source quotes;
- tolerate harmless whitespace differences;
- reject fabricated quotes;
- reject partial/mismatched text when exact verification is required;
- provide a diagnostic similarity score for non-exact candidates;
- preserve source traceability.

The tests intentionally use small transcript snippets so they remain
fast and deterministic.
"""

from __future__ import annotations

import pytest

from src.evidence.quote_verifier import QuoteVerifier


@pytest.fixture
def verifier() -> QuoteVerifier:
    """Create a quote verifier instance."""
    return QuoteVerifier()


@pytest.fixture
def source_text() -> str:
    """Return representative source transcript text."""
    return (
        "The biggest barrier is capital budget approval. "
        "Hospitals need a clear economic case. "
        "Finance also wants to understand utilisation, procedure volume, "
        "maintenance cost, and payback."
    )


def test_exact_quote_is_verified(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """An exact quote from the source should pass verification."""
    quote = "The biggest barrier is capital budget approval."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True


def test_exact_quote_returns_success_status(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """Successful verification should expose a positive verification status."""
    quote = "Hospitals need a clear economic case."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True
    assert result.status.value.lower() in {
        "verified",
        "exact",
        "passed",
    }


def test_fabricated_quote_is_rejected(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """A quote not present in the transcript must not be verified."""
    quote = "Robotic surgery has a 40% adoption rate in France."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is False


def test_whitespace_difference_does_not_break_verification(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """Harmless whitespace differences should still match the source."""
    quote = "The   biggest   barrier is   capital budget approval."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True


def test_leading_and_trailing_whitespace_is_ignored(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """Leading/trailing whitespace should not invalidate an exact quote."""
    quote = "   The biggest barrier is capital budget approval.   "

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True


def test_case_difference_is_handled(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """Quote verification should tolerate case-only differences."""
    quote = "THE BIGGEST BARRIER IS CAPITAL BUDGET APPROVAL."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True


def test_empty_quote_is_rejected(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """An empty quote cannot be considered verified."""
    result = verifier.verify(
        quote="",
        source_text=source_text,
    )

    assert result.is_verified is False


def test_empty_source_is_rejected(
    verifier: QuoteVerifier,
) -> None:
    """A quote cannot be verified without source text."""
    result = verifier.verify(
        quote="The biggest barrier is capital budget approval.",
        source_text="",
    )

    assert result.is_verified is False


def test_empty_quote_and_empty_source_are_rejected(
    verifier: QuoteVerifier,
) -> None:
    """Both empty inputs should fail safely."""
    result = verifier.verify(
        quote="",
        source_text="",
    )

    assert result.is_verified is False


def test_quote_from_later_source_section_is_verified(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """A valid quote should be found regardless of its source position."""
    quote = "maintenance cost, and payback."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True


def test_quote_with_punctuation_difference(
    verifier: QuoteVerifier,
) -> None:
    """Minor punctuation differences should not cause false rejection."""
    source = "Finance wants to understand utilisation, procedure volume, and payback."

    quote = "Finance wants to understand utilisation procedure volume and payback"

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is True


def test_substantially_different_quote_is_rejected(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """A semantically related but fabricated quote must be rejected."""
    quote = (
        "Finance requires hospitals to achieve a guaranteed return "
        "within twelve months."
    )

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is False


def test_similarity_is_available_for_non_exact_quote(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """The verifier should expose a diagnostic similarity when possible."""
    quote = "The main barrier is hospital capital approval."

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is False

    if result.similarity is not None:
        assert 0.0 <= result.similarity <= 1.0


def test_similarity_is_high_for_near_match(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """A near-match should normally receive a stronger similarity signal."""
    quote = "The biggest barrier is capital budget approval"

    result = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert result.is_verified is True

    if result.similarity is not None:
        assert result.similarity >= 0.9


def test_quote_verification_is_deterministic(
    verifier: QuoteVerifier,
    source_text: str,
) -> None:
    """Repeated verification of the same inputs should be deterministic."""
    quote = "Hospitals need a clear economic case."

    first = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    second = verifier.verify(
        quote=quote,
        source_text=source_text,
    )

    assert first.is_verified == second.is_verified
    assert first.status == second.status
    assert first.similarity == second.similarity


def test_quote_is_verified_when_source_contains_multiple_sentences(
    verifier: QuoteVerifier,
) -> None:
    """Exact quotes should work inside a larger transcript passage."""
    source = (
        "Adoption is growing. "
        "It is concentrated in larger academic centres. "
        "Smaller regional hospitals are moving more slowly. "
        "Capital budget approval remains important."
    )

    quote = "Smaller regional hospitals are moving more slowly."

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is True


def test_fabricated_percentage_is_not_verified(
    verifier: QuoteVerifier,
) -> None:
    """
    Numeric claims not present in the source must not become exact quotes.

    This is particularly important for the Hasamex case because the
    transcripts contain qualitative and some growth-rate statements but
    do not establish arbitrary market-share percentages.
    """
    source = (
        "Adoption is growing, particularly in larger academic centres. "
        "Smaller regional hospitals are slower."
    )

    quote = "France currently has 35% robotic surgery market share."

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is False


def test_verifier_does_not_accept_source_substring_as_wrong_quote(
    verifier: QuoteVerifier,
) -> None:
    """
    A quote containing fabricated text around a valid phrase must fail.

    The verifier should not accept a quote simply because one small
    portion of it exists in the source.
    """
    source = "The biggest barrier is capital budget approval."

    quote = (
        "The biggest barrier is capital budget approval, "
        "and France has the highest market share."
    )

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is False


def test_exact_quote_preserves_original_words(
    verifier: QuoteVerifier,
) -> None:
    """The verified quote should correspond to actual source wording."""
    source = (
        "Training matters especially in the first year; "
        "several surgeons should be trained."
    )

    quote = "Training matters especially in the first year"

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is True


def test_quote_verification_does_not_infer_missing_information(
    verifier: QuoteVerifier,
) -> None:
    """
    Semantic plausibility must not be treated as exact quote evidence.
    """
    source = (
        "ROI is very important. "
        "Finance wants utilisation, procedure volume, maintenance cost, "
        "and payback."
    )

    quote = "ROI is important because hospitals need positive financial returns."

    result = verifier.verify(
        quote=quote,
        source_text=source,
    )

    assert result.is_verified is False