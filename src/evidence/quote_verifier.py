"""
Deterministic quote verification for Hasamex Expert Analysis.

The verifier compares candidate quotes against original transcript text.
It never generates or semantically invents quotation text.

Supported behavior:
- exact quote verification
- case-insensitive matching
- punctuation-insensitive matching
- whitespace normalization
- Unicode punctuation normalization
- deterministic similarity scoring
- enum-based verification status
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# STATUS
# ======================================================================


class QuoteVerificationStatus(str, Enum):
    """Stable verification states exposed by the quote verifier."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


# ======================================================================
# RESULT
# ======================================================================


@dataclass(frozen=True)
class QuoteVerificationResult:
    """
    Result returned by quote verification.

    `status` is an Enum rather than a plain string so callers can safely
    use both:

        result.status == QuoteVerificationStatus.VERIFIED

    and:

        result.status.value == "verified"
    """

    verified: bool
    similarity: float
    confidence: float
    matched_text: Optional[str] = None
    reason: str = ""

    @property
    def status(self) -> QuoteVerificationStatus:
        """Return the enum representation of the verification state."""

        if self.verified:
            return QuoteVerificationStatus.VERIFIED

        return QuoteVerificationStatus.UNVERIFIED

    @property
    def is_verified(self) -> bool:
        """Backward-compatible boolean accessor."""

        return self.verified


# ======================================================================
# NORMALIZATION
# ======================================================================


def _unicode_normalize(text: str) -> str:
    """Normalize Unicode and common typographic characters."""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    return text


def normalize_quote(text: str) -> str:
    """
    Normalize text for deterministic quote comparison.

    The operation:
    - Unicode-normalizes the input
    - converts to lowercase
    - removes ASCII punctuation
    - removes Unicode punctuation
    - collapses whitespace

    Semantic paraphrasing is intentionally NOT performed.
    """

    if text is None:
        return ""

    normalized = _unicode_normalize(str(text))
    normalized = normalized.lower()

    normalized = normalized.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


# ======================================================================
# INTERNAL MATCHING
# ======================================================================


def _build_normalized_character_map(
    source: str,
) -> Tuple[str, List[int]]:
    """
    Build normalized source text and map normalized characters back to
    positions in the original source.
    """

    normalized_parts: List[str] = []
    positions: List[int] = []

    for index, character in enumerate(source):
        normalized = normalize_quote(character)

        if not normalized:
            continue

        for normalized_character in normalized:
            normalized_parts.append(normalized_character)
            positions.append(index)

    return "".join(normalized_parts), positions


def _find_normalized_substring(
    source: str,
    candidate: str,
) -> Optional[str]:
    """
    Find a candidate inside source after deterministic normalization.

    Returns the original source substring when found.
    """

    normalized_source, positions = (
        _build_normalized_character_map(source)
    )

    normalized_candidate = normalize_quote(candidate)

    if not normalized_source:
        return None

    if not normalized_candidate:
        return None

    start = normalized_source.find(
        normalized_candidate
    )

    if start < 0:
        return None

    end = start + len(normalized_candidate) - 1

    if start >= len(positions):
        return None

    if end >= len(positions):
        return None

    original_start = positions[start]
    original_end = positions[end]

    return source[
        original_start : original_end + 1
    ]


def _best_similarity(
    candidate: str,
    source: str,
) -> float:
    """
    Calculate deterministic similarity between candidate and source.

    Exact normalized containment receives 1.0.

    For non-exact candidates, SequenceMatcher is used against the full
    source and deterministic local windows.
    """

    candidate_normalized = normalize_quote(candidate)
    source_normalized = normalize_quote(source)

    if not candidate_normalized:
        return 0.0

    if not source_normalized:
        return 0.0

    if candidate_normalized in source_normalized:
        return 1.0

    candidate_length = len(candidate_normalized)

    best = SequenceMatcher(
        None,
        candidate_normalized,
        source_normalized,
        autojunk=False,
    ).ratio()

    window_sizes = {
        candidate_length,
        max(
            1,
            int(candidate_length * 0.90),
        ),
        max(
            1,
            int(candidate_length * 1.10),
        ),
    }

    for window_size in sorted(window_sizes):
        if window_size >= len(source_normalized):
            continue

        step = max(
            1,
            window_size // 4,
        )

        for start in range(
            0,
            len(source_normalized) - window_size + 1,
            step,
        ):
            window = source_normalized[
                start : start + window_size
            ]

            ratio = SequenceMatcher(
                None,
                candidate_normalized,
                window,
                autojunk=False,
            ).ratio()

            if ratio > best:
                best = ratio

    return max(
        0.0,
        min(
            1.0,
            float(best),
        ),
    )


# ======================================================================
# VERIFIER
# ======================================================================


class QuoteVerifier:
    """Deterministic transcript quote verifier."""

    DEFAULT_SIMILARITY_THRESHOLD = 0.80

    def __init__(
        self,
        similarity_threshold: float = (
            DEFAULT_SIMILARITY_THRESHOLD
        ),
    ) -> None:

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        self.similarity_threshold = float(
            similarity_threshold
        )

    def verify(
        self,
        quote: str,
        source_text: str,
    ) -> QuoteVerificationResult:
        """Verify a candidate quote against source transcript text."""

        if quote is None:
            return QuoteVerificationResult(
                verified=False,
                similarity=0.0,
                confidence=0.0,
                matched_text=None,
                reason="Quote is missing.",
            )

        if source_text is None:
            return QuoteVerificationResult(
                verified=False,
                similarity=0.0,
                confidence=0.0,
                matched_text=None,
                reason="Source text is missing.",
            )

        candidate = str(quote).strip()
        source = str(source_text)

        if not candidate:
            return QuoteVerificationResult(
                verified=False,
                similarity=0.0,
                confidence=0.0,
                matched_text=None,
                reason="Quote is empty.",
            )

        if not source.strip():
            return QuoteVerificationResult(
                verified=False,
                similarity=0.0,
                confidence=0.0,
                matched_text=None,
                reason="Source text is empty.",
            )

        # --------------------------------------------------------------
        # Exact normalized match
        # --------------------------------------------------------------

        matched_text = _find_normalized_substring(
            source,
            candidate,
        )

        if matched_text is not None:
            return QuoteVerificationResult(
                verified=True,
                similarity=1.0,
                confidence=1.0,
                matched_text=matched_text,
                reason=(
                    "Quote verified after case, punctuation, "
                    "Unicode, and whitespace normalization."
                ),
            )

        # --------------------------------------------------------------
        # Similarity fallback
        # --------------------------------------------------------------

        similarity = _best_similarity(
            candidate,
            source,
        )

        verified = (
            similarity
            >= self.similarity_threshold
        )

        if verified:
            reason = (
                "Quote is a high-similarity match to the "
                "source transcript."
            )
        else:
            reason = (
                "Candidate quote could not be verified as an "
                "exact normalized source quote."
            )

        return QuoteVerificationResult(
            verified=verified,
            similarity=similarity,
            confidence=similarity,
            matched_text=None,
            reason=reason,
        )

    def verify_quote(
        self,
        quote: str,
        source_text: str,
    ) -> QuoteVerificationResult:
        """Backward-compatible verification alias."""

        return self.verify(
            quote=quote,
            source_text=source_text,
        )


# ======================================================================
# FUNCTIONAL API
# ======================================================================


def verify_quote(
    quote: str,
    source_text: str,
    similarity_threshold: float = (
        QuoteVerifier.DEFAULT_SIMILARITY_THRESHOLD
    ),
) -> QuoteVerificationResult:
    """Verify one quote against one transcript source."""

    verifier = QuoteVerifier(
        similarity_threshold=similarity_threshold
    )

    return verifier.verify(
        quote=quote,
        source_text=source_text,
    )


def verify_quote_against_sources(
    quote: str,
    sources: Sequence[str],
    similarity_threshold: float = (
        QuoteVerifier.DEFAULT_SIMILARITY_THRESHOLD
    ),
) -> QuoteVerificationResult:
    """Verify a quote against multiple transcript sources."""

    if not sources:
        return QuoteVerificationResult(
            verified=False,
            similarity=0.0,
            confidence=0.0,
            matched_text=None,
            reason="No source texts were supplied.",
        )

    verifier = QuoteVerifier(
        similarity_threshold=similarity_threshold
    )

    best_result: Optional[
        QuoteVerificationResult
    ] = None

    for source in sources:
        result = verifier.verify(
            quote=quote,
            source_text=source,
        )

        if best_result is None:
            best_result = result
            continue

        if result.similarity > best_result.similarity:
            best_result = result

    assert best_result is not None

    return best_result


def verify_quotes(
    quotes: Iterable[str],
    source_text: str,
    similarity_threshold: float = (
        QuoteVerifier.DEFAULT_SIMILARITY_THRESHOLD
    ),
) -> List[QuoteVerificationResult]:
    """Verify multiple quotes against one transcript."""

    verifier = QuoteVerifier(
        similarity_threshold=similarity_threshold
    )

    return [
        verifier.verify(
            quote=quote,
            source_text=source_text,
        )
        for quote in quotes
    ]