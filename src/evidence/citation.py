"""
Citation construction for source-grounded transcript evidence.

This module converts verified evidence into human-readable and
machine-readable citations.

Citation design goals:
    - Every citation identifies the expert.
    - Every citation identifies the source transcript.
    - Every citation contains the original timestamp.
    - Exact quotes are only marked as quotes when verified.
    - Citation generation never invents timestamps.
    - Citation objects remain compatible with the application's
      Pydantic data models.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from exception import SensorException
from logger import logging

from src.models import Citation, Evidence, EvidenceStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationContext:
    """
    Optional context used when rendering citations.
    """

    include_expert: bool = True
    include_market: bool = True
    include_source_file: bool = True
    include_timestamp: bool = True


class CitationBuilder:
    """
    Build traceable citations from evidence objects.

    The builder does not generate or infer missing source metadata.
    Missing timestamps remain missing rather than being guessed.
    """

    def __init__(
        self,
        context: CitationContext | None = None,
    ) -> None:
        """
        Initialize the citation builder.

        Args:
            context: Controls which metadata fields are rendered in
                human-readable citation labels.
        """
        self.context = context or CitationContext()

    def build(self, evidence: Evidence) -> Citation:
        """
        Convert one Evidence object into a Citation.

        Args:
            evidence: Evidence containing source metadata.

        Returns:
            A structured Citation model.

        Raises:
            SensorException: If citation construction fails.
        """
        try:
            self._validate_evidence(evidence)

            label = self._build_label(evidence)

            return Citation(
                citation_id=self._build_citation_id(evidence),
                document_id=evidence.document_id,
                source_file=evidence.source_file,
                expert_name=evidence.expert_name,
                market=evidence.market,
                start_timestamp=evidence.start_timestamp,
                end_timestamp=evidence.end_timestamp,
                label=label,
            )

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to build citation for evidence."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def build_many(
        self,
        evidence_items: Iterable[Evidence],
    ) -> list[Citation]:
        """
        Build citations for multiple evidence items.

        Duplicate citations are removed while preserving order.
        """
        try:
            citations: list[Citation] = []
            seen: set[str] = set()

            for evidence in evidence_items:
                citation = self.build(evidence)

                if citation.citation_id in seen:
                    continue

                seen.add(citation.citation_id)
                citations.append(citation)

            return citations

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to build multiple citations."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def render(self, citation: Citation) -> str:
        """
        Render a structured citation as a human-readable string.

        Example:

            Dr. Jean Martin | France | Transcript_1_France.txt | 01:20

        If an end timestamp exists:

            Dr. Jean Martin | France | Transcript_1_France.txt | 01:20–01:55
        """
        try:
            parts: list[str] = []

            if self.context.include_expert and citation.expert_name:
                parts.append(citation.expert_name)

            if self.context.include_market and citation.market:
                parts.append(citation.market)

            if (
                self.context.include_source_file
                and citation.source_file
            ):
                parts.append(citation.source_file)

            timestamp = self._format_timestamp_range(
                citation.start_timestamp,
                citation.end_timestamp,
            )

            if self.context.include_timestamp and timestamp:
                parts.append(timestamp)

            if not parts:
                return "Source unavailable"

            return " | ".join(parts)

        except Exception as error:
            logger.exception(
                "Failed to render citation."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def render_many(
        self,
        citations: Iterable[Citation],
    ) -> list[str]:
        """
        Render multiple citations in input order.
        """
        try:
            return [
                self.render(citation)
                for citation in citations
            ]

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to render multiple citations."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def render_markdown(
        self,
        citation: Citation,
    ) -> str:
        """
        Render a citation in a compact Markdown-friendly format.

        Example:

            **Dr. Jean Martin** · France ·
            `Transcript_1_France.txt` · **01:20**
        """
        try:
            expert = (
                f"**{citation.expert_name}**"
                if citation.expert_name
                else ""
            )

            market = (
                citation.market
                if citation.market
                else ""
            )

            source = (
                f"`{citation.source_file}`"
                if citation.source_file
                else ""
            )

            timestamp = self._format_timestamp_range(
                citation.start_timestamp,
                citation.end_timestamp,
            )

            timestamp_text = (
                f"**{timestamp}**"
                if timestamp
                else ""
            )

            parts = [
                value
                for value in (
                    expert,
                    market,
                    source,
                    timestamp_text,
                )
                if value
            ]

            if not parts:
                return "Source unavailable"

            return " · ".join(parts)

        except Exception as error:
            logger.exception(
                "Failed to render Markdown citation."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def _build_label(self, evidence: Evidence) -> str:
        """
        Build the canonical citation label.
        """
        parts: list[str] = []

        if self.context.include_expert and evidence.expert_name:
            parts.append(evidence.expert_name)

        if self.context.include_market and evidence.market:
            parts.append(evidence.market)

        if (
            self.context.include_source_file
            and evidence.source_file
        ):
            parts.append(evidence.source_file)

        timestamp = self._format_timestamp_range(
            evidence.start_timestamp,
            evidence.end_timestamp,
        )

        if self.context.include_timestamp and timestamp:
            parts.append(timestamp)

        if not parts:
            return "Source unavailable"

        return " | ".join(parts)

    @staticmethod
    def _build_citation_id(
        evidence: Evidence,
    ) -> str:
        """
        Build a deterministic citation identifier.

        The identifier is based exclusively on source metadata. It does
        not contain generated answer text.
        """
        components = [
            evidence.document_id,
            evidence.source_file,
            evidence.expert_name,
            evidence.start_timestamp,
            evidence.end_timestamp or "",
        ]

        normalized = "|".join(
            component.strip()
            for component in components
        )

        return f"citation:{normalized}"

    @staticmethod
    def _format_timestamp_range(
        start_timestamp: str | None,
        end_timestamp: str | None,
    ) -> str:
        """
        Format an existing timestamp range.

        No timestamp is generated when source metadata does not provide one.
        """
        if not start_timestamp:
            return ""

        if (
            end_timestamp
            and end_timestamp != start_timestamp
        ):
            return f"{start_timestamp}–{end_timestamp}"

        return start_timestamp

    @staticmethod
    def _validate_evidence(
        evidence: Evidence,
    ) -> None:
        """
        Validate that evidence contains the minimum source information
        required to construct a useful citation.
        """
        if not evidence.document_id.strip():
            raise ValueError(
                "Evidence document_id cannot be empty."
            )

        if not evidence.source_file.strip():
            raise ValueError(
                "Evidence source_file cannot be empty."
            )

        if not evidence.expert_name.strip():
            raise ValueError(
                "Evidence expert_name cannot be empty."
            )

        if not evidence.quote.strip():
            raise ValueError(
                "Evidence quote cannot be empty."
            )

        if evidence.status == EvidenceStatus.VERIFIED:
            if not evidence.start_timestamp:
                raise ValueError(
                    "Verified evidence must contain a start timestamp."
                )


def build_citation(
    evidence: Evidence,
) -> Citation:
    """
    Convenience function for building one citation.
    """
    return CitationBuilder().build(evidence)


def build_citations(
    evidence_items: Iterable[Evidence],
) -> list[Citation]:
    """
    Convenience function for building multiple citations.
    """
    return CitationBuilder().build_many(evidence_items)


def render_citation(
    citation: Citation,
) -> str:
    """
    Convenience function for rendering one citation.
    """
    return CitationBuilder().render(citation)


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "CitationBuilder",
    "CitationContext",
    "build_citation",
    "build_citations",
    "render_citation",
]