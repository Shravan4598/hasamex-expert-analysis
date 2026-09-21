"""
Exact quote verification against original transcript sources.

This module validates whether a proposed quote is actually present in
the original transcript text.

Important design principle:

    Retrieval text is not treated as the final authority for quotes.

The original transcript source is authoritative. A quote can only be
marked as VERIFIED when its normalized form can be located in the
original source.

The verifier supports:
    - exact matching
    - whitespace normalization
    - Unicode punctuation normalization
    - quote extraction from transcript segments
    - verification against specific segments
    - verification against complete transcript documents

It never invents or repairs a quote when verification fails.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from exception import SensorException
from logger import logging

from src.models import EvidenceStatus, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuoteVerificationResult:
    """
    Result of validating a proposed quote against source text.
    """

    quote: str
    verified: bool
    status: EvidenceStatus
    matched_text: str | None = None
    start_character: int | None = None
    end_character: int | None = None
    confidence: float = 0.0
    message: str = ""


class QuoteVerifier:
    """
    Verify exact quotes against authoritative transcript text.

    The verifier is intentionally conservative. Semantic similarity is
    never sufficient to mark a quote as exact.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.96,
        minimum_quote_length: int = 3,
    ) -> None:
        """
        Initialize the quote verifier.

        Args:
            fuzzy_threshold: Similarity threshold used only for diagnostics.
                Fuzzy matches are NEVER automatically marked as exact quotes.
            minimum_quote_length: Minimum number of non-whitespace characters
                required for quote verification.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError(
                "fuzzy_threshold must be between 0.0 and 1.0."
            )

        if minimum_quote_length <= 0:
            raise ValueError(
                "minimum_quote_length must be greater than zero."
            )

        self.fuzzy_threshold = fuzzy_threshold
        self.minimum_quote_length = minimum_quote_length

    def verify(
        self,
        quote: str,
        source_text: str,
    ) -> QuoteVerificationResult:
        """
        Verify a proposed quote against the original source text.

        Exact verification tolerates formatting differences such as:
            - repeated whitespace
            - line breaks
            - Unicode normalization
            - curly versus straight apostrophes

        It does NOT tolerate changed words.

        Args:
            quote: Proposed exact quote.
            source_text: Authoritative original transcript text.

        Returns:
            QuoteVerificationResult describing verification status.
        """
        try:
            quote = self._validate_quote(quote)
            source_text = self._validate_source(source_text)

            normalized_quote = self.normalize_text(quote)
            normalized_source = self.normalize_text(source_text)

            match_start, match_end = self._find_normalized_match(
                normalized_quote,
                normalized_source,
            )

            if match_start is not None:
                matched_text = normalized_source[
                    match_start:match_end
                ]

                return QuoteVerificationResult(
                    quote=quote,
                    verified=True,
                    status=EvidenceStatus.VERIFIED,
                    matched_text=matched_text,
                    start_character=match_start,
                    end_character=match_end,
                    confidence=1.0,
                    message="Quote verified against the original source.",
                )

            diagnostic_similarity = self._similarity(
                normalized_quote,
                normalized_source,
            )

            return QuoteVerificationResult(
                quote=quote,
                verified=False,
                status=EvidenceStatus.UNVERIFIED,
                confidence=diagnostic_similarity,
                message=(
                    "The proposed quote was not found verbatim in the "
                    "original source. It must not be displayed as an "
                    "exact quote."
                ),
            )

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Quote verification failed.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def verify_against_segments(
        self,
        quote: str,
        segments: Iterable[TranscriptSegment],
    ) -> QuoteVerificationResult:
        """
        Verify a quote against a collection of transcript segments.

        This method is useful when the exact original document text is not
        directly available but the parser has preserved source segments.

        Args:
            quote: Proposed quote.
            segments: Source transcript segments.

        Returns:
            Verification result.
        """
        try:
            segment_list = list(segments)

            if not segment_list:
                raise ValueError(
                    "At least one transcript segment is required."
                )

            combined_source = "\n".join(
                segment.text
                for segment in segment_list
                if segment.text.strip()
            )

            result = self.verify(
                quote=quote,
                source_text=combined_source,
            )

            if result.verified:
                return result

            # A quote may correspond to a single segment even when
            # surrounding formatting differs. Check each source segment
            # independently before returning failure.
            for segment in segment_list:
                segment_result = self.verify(
                    quote=quote,
                    source_text=segment.text,
                )

                if segment_result.verified:
                    return segment_result

            return result

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Segment-level quote verification failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def verify_multiple(
        self,
        quotes: Iterable[str],
        source_text: str,
    ) -> list[QuoteVerificationResult]:
        """
        Verify multiple quotes against the same source document.

        Args:
            quotes: Proposed quotes.
            source_text: Authoritative source text.

        Returns:
            Verification results in input order.
        """
        try:
            return [
                self.verify(
                    quote=quote,
                    source_text=source_text,
                )
                for quote in quotes
            ]

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Multiple quote verification failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def find_exact_quote(
        self,
        quote: str,
        source_text: str,
    ) -> str | None:
        """
        Return the exact source substring corresponding to a verified quote.

        This is useful when the source contains formatting differences and
        the UI should display the original source wording.

        Args:
            quote: Proposed quote.
            source_text: Original transcript.

        Returns:
            Exact source substring when found, otherwise None.
        """
        try:
            quote = self._validate_quote(quote)
            source_text = self._validate_source(source_text)

            normalized_quote = self.normalize_text(quote)

            mapping = self._build_normalized_source_mapping(
                source_text
            )

            normalized_source = mapping.normalized_text

            normalized_position = normalized_source.find(
                normalized_quote
            )

            if normalized_position == -1:
                return None

            normalized_end = (
                normalized_position + len(normalized_quote)
            )

            original_start = mapping.normalized_to_original(
                normalized_position
            )

            original_end = mapping.normalized_to_original_end(
                normalized_end
            )

            if original_start is None or original_end is None:
                return None

            return source_text[
                original_start:original_end
            ]

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to locate exact source quote."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for quote comparison.

        Normalization removes formatting differences while preserving
        lexical content.

        Transformations include:
            - Unicode normalization.
            - Curly quote normalization.
            - Dash normalization.
            - Whitespace collapsing.
            - Leading/trailing whitespace removal.

        Words are never paraphrased or replaced.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Text must be a string, got {type(text).__name__}."
            )

        normalized = unicodedata.normalize(
            "NFKC",
            text,
        )

        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u00a0": " ",
        }

        for source, target in replacements.items():
            normalized = normalized.replace(
                source,
                target,
            )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    def is_safe_exact_quote(
        self,
        quote: str,
        source_text: str,
    ) -> bool:
        """
        Convenience method returning whether a quote is verified exactly.
        """
        return self.verify(
            quote=quote,
            source_text=source_text,
        ).verified

    def _find_normalized_match(
        self,
        normalized_quote: str,
        normalized_source: str,
    ) -> tuple[int | None, int | None]:
        """
        Find a normalized quote in normalized source text.
        """
        if not normalized_quote:
            return None, None

        position = normalized_source.find(
            normalized_quote
        )

        if position == -1:
            return None, None

        return (
            position,
            position + len(normalized_quote),
        )

    def _similarity(
        self,
        quote: str,
        source: str,
    ) -> float:
        """
        Calculate diagnostic similarity.

        This value is informational only. It must never be used to
        label a quote as verified.
        """
        if not quote or not source:
            return 0.0

        # Comparing the quote against the entire source is only a
        # diagnostic fallback. Exact verification always uses substring
        # matching.
        if len(quote) <= len(source):
            best_ratio = 0.0

            window_length = len(quote)

            for start in range(
                0,
                len(source) - window_length + 1,
            ):
                window = source[
                    start:start + window_length
                ]

                ratio = SequenceMatcher(
                    None,
                    quote,
                    window,
                ).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio

                    if best_ratio >= self.fuzzy_threshold:
                        break

            return best_ratio

        return SequenceMatcher(
            None,
            quote,
            source,
        ).ratio()

    def _validate_quote(self, quote: str) -> str:
        """
        Validate a proposed quote.
        """
        if not isinstance(quote, str):
            raise TypeError(
                f"Quote must be a string, got {type(quote).__name__}."
            )

        cleaned = quote.strip()

        if len(re.sub(r"\s+", "", cleaned)) < self.minimum_quote_length:
            raise ValueError(
                "Quote is too short to verify reliably."
            )

        return cleaned

    @staticmethod
    def _validate_source(source_text: str) -> str:
        """
        Validate authoritative source text.
        """
        if not isinstance(source_text, str):
            raise TypeError(
                "source_text must be a string."
            )

        if not source_text.strip():
            raise ValueError(
                "source_text cannot be empty."
            )

        return source_text

    def _build_normalized_source_mapping(
        self,
        source_text: str,
    ) -> "_NormalizedSourceMapping":
        """
        Build a mapping between normalized and original source positions.

        This allows find_exact_quote() to return the original source
        substring rather than the normalized representation.
        """
        normalized_chars: list[str] = []
        position_mapping: list[int] = []

        previous_was_space = False

        for original_index, character in enumerate(source_text):
            normalized_character = unicodedata.normalize(
                "NFKC",
                character,
            )

            replacement_map = {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\u00a0": " ",
            }

            normalized_character = replacement_map.get(
                normalized_character,
                normalized_character,
            )

            if normalized_character.isspace():
                if previous_was_space:
                    continue

                normalized_chars.append(" ")
                position_mapping.append(original_index)
                previous_was_space = True
                continue

            normalized_chars.append(normalized_character)
            position_mapping.append(original_index)
            previous_was_space = False

        normalized_text = "".join(normalized_chars).strip()

        # The strip() operation may remove mapped positions at the
        # beginning/end, so rebuild the mapping for the stripped text.
        leading_spaces = len(normalized_text) - len(
            normalized_text.lstrip()
        )

        if leading_spaces:
            normalized_text = normalized_text.lstrip()
            position_mapping = position_mapping[leading_spaces:]

        trailing_spaces = len(normalized_text) - len(
            normalized_text.rstrip()
        )

        if trailing_spaces:
            normalized_text = normalized_text.rstrip()
            position_mapping = position_mapping[:-trailing_spaces]

        return _NormalizedSourceMapping(
            normalized_text=normalized_text,
            position_mapping=position_mapping,
        )


@dataclass(frozen=True)
class _NormalizedSourceMapping:
    """
    Internal mapping from normalized text positions to original positions.
    """

    normalized_text: str
    position_mapping: list[int]

    def normalized_to_original(
        self,
        normalized_position: int,
    ) -> int | None:
        """
        Convert a normalized character position to an original position.
        """
        if not self.position_mapping:
            return None

        if normalized_position < 0:
            return None

        if normalized_position >= len(self.position_mapping):
            return None

        return self.position_mapping[normalized_position]

    def normalized_to_original_end(
        self,
        normalized_end: int,
    ) -> int | None:
        """
        Convert a normalized exclusive end position to an original
        exclusive end position.
        """
        if not self.position_mapping:
            return None

        if normalized_end <= 0:
            return None

        if normalized_end > len(self.position_mapping):
            normalized_end = len(self.position_mapping)

        last_position = self.position_mapping[
            normalized_end - 1
        ]

        return last_position + 1


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "QuoteVerificationResult",
    "QuoteVerifier",
]