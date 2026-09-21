"""
Cross-expert theme analysis for the Hasamex Expert Analysis application.

This module identifies recurring themes across expert transcripts.

Important design principles:

    - Themes are derived from transcript evidence.
    - The LLM is not allowed to invent supporting evidence.
    - Every theme must retain source-grounded evidence.
    - Expert names and timestamps come from application metadata.
    - Exact quotes are verified before being exposed as verified quotes.
    - The module distinguishes broad recurring themes from unsupported
      generalizations.

The intended workflow is:

    transcripts
        -> retrieval
        -> reranking
        -> evidence selection
        -> grounded LLM synthesis
        -> quote verification
        -> citation construction
        -> Theme objects
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from exception import SensorException
from logger import logging
from src.analysis.llm import GeminiLLMService, get_llm_service
from src.config import Settings, get_settings
from src.evidence.citation import CitationBuilder
from src.evidence.quote_verifier import QuoteVerifier
from src.models import (
    Evidence,
    Theme,
    ThemeEvidence,
)
from src.retrieval.reranker import RetrievalReranker
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThemeAnalysisConfig:
    """
    Configuration for cross-expert theme analysis.
    """

    retrieval_top_k: int
    rerank_top_k: int
    max_themes: int
    minimum_experts: int
    minimum_evidence: int


class ThemeAnalyzer:
    """
    Identify common themes across multiple expert transcripts.

    The analyzer intentionally uses a two-stage approach:

        Stage 1:
            Retrieve relevant evidence from the transcripts.

        Stage 2:
            Ask the LLM to synthesize recurring themes from only that
            retrieved evidence.

    This prevents the model from being asked to discover themes from
    unrestricted transcript text without source controls.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: RetrievalReranker,
        llm_service: GeminiLLMService | None = None,
        quote_verifier: QuoteVerifier | None = None,
        citation_builder: CitationBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the theme analyzer.
        """
        try:
            self.settings = settings or get_settings()

            self.retriever = retriever
            self.reranker = reranker
            self.llm_service = llm_service or get_llm_service()
            self.quote_verifier = (
                quote_verifier or QuoteVerifier()
            )
            self.citation_builder = (
                citation_builder or CitationBuilder()
            )

            self.config = ThemeAnalysisConfig(
                retrieval_top_k=self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
                max_themes=10,
                minimum_experts=2,
                minimum_evidence=2,
            )

        except Exception as error:
            logger.exception(
                "Failed to initialize ThemeAnalyzer."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def analyze(
        self,
        *,
        focus: str | None = None,
        expert_names: Iterable[str] | None = None,
        max_themes: int | None = None,
    ) -> list[Theme]:
        """
        Identify recurring themes across the available transcripts.

        Args:
            focus: Optional topic to focus the theme analysis on.
            expert_names: Optional expert scope.
            max_themes: Maximum number of themes to return.

        Returns:
            List of grounded Theme objects.
        """
        try:
            normalized_experts = self._normalize_experts(
                expert_names
            )

            query = self._build_theme_query(
                focus=focus
            )

            retrieval_results = self.retriever.retrieve(
                query=query,
                top_k=self.config.retrieval_top_k,
            )

            reranked_results = self.reranker.rerank_by_expert_balance(
                query=query,
                results=retrieval_results,
                top_k=self.config.rerank_top_k,
            )

            evidence = self._results_to_evidence(
                reranked_results
            )

            if normalized_experts:
                evidence = [
                    item
                    for item in evidence
                    if item.expert_name in normalized_experts
                ]

            evidence = self._verify_evidence(
                evidence
            )

            if len(evidence) < self.config.minimum_evidence:
                logger.warning(
                    "Insufficient evidence for theme analysis."
                )
                return []

            if len(
                {
                    item.expert_name
                    for item in evidence
                    if item.expert_name
                }
            ) < self.config.minimum_experts:
                logger.warning(
                    "Theme analysis requires evidence from at least "
                    "%s experts.",
                    self.config.minimum_experts,
                )
                return []

            raw_themes = self._generate_themes(
                query=query,
                evidence=evidence,
                max_themes=max_themes or self.config.max_themes,
            )

            return self._build_theme_objects(
                raw_themes=raw_themes,
                evidence=evidence,
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Cross-expert theme analysis failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def analyze_for_question(
        self,
        question: str,
        *,
        expert_names: Iterable[str] | None = None,
        max_themes: int | None = None,
    ) -> list[Theme]:
        """
        Identify themes specifically related to an interview question.
        """
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        return self.analyze(
            focus=question,
            expert_names=expert_names,
            max_themes=max_themes,
        )

    def _build_theme_query(
        self,
        focus: str | None,
    ) -> str:
        """
        Build a retrieval query for theme discovery.
        """
        if focus and focus.strip():
            return (
                f"Identify recurring expert perspectives, common themes, "
                f"shared observations, and differences related to: "
                f"{focus.strip()}"
            )

        return (
            "Identify recurring themes, common observations, shared "
            "concerns, adoption drivers, barriers, economics, training, "
            "clinical outcomes, future trends, and purchasing processes "
            "across the expert interviews."
        )

    def _generate_themes(
        self,
        query: str,
        evidence: list[Evidence],
        max_themes: int,
    ) -> list[dict[str, Any]]:
        """
        Ask the grounded LLM to identify recurring themes.

        The LLM returns only theme descriptions and references to
        supplied evidence indices. Source metadata is subsequently
        resolved by the application.
        """
        prompt = self._build_theme_prompt(
            query=query,
            evidence=evidence,
            max_themes=max_themes,
        )

        response = self.llm_service.generate_json(
            prompt=prompt,
            schema=self._theme_schema(),
        )

        themes = response.get(
            "themes",
            [],
        )

        if not isinstance(themes, list):
            return []

        return [
            item
            for item in themes
            if isinstance(item, dict)
        ]

    def _build_theme_prompt(
        self,
        query: str,
        evidence: list[Evidence],
        max_themes: int,
    ) -> str:
        """
        Build the grounded theme-analysis prompt.
        """
        evidence_block = self._serialize_evidence(
            evidence
        )

        return f"""
Identify common themes across the supplied expert interview evidence.

FOCUS:
{query}

MAXIMUM THEMES:
{max_themes}

STRICT RULES:

1. Use ONLY the evidence supplied below.
2. A theme should be supported by statements from at least two
   different experts.
3. Do not invent expert opinions.
4. Do not invent quotes.
5. Do not invent timestamps.
6. Do not infer unsupported market facts.
7. If experts discuss a topic differently, do not incorrectly label
   it as a shared agreement.
8. Prefer concrete themes such as:
      - adoption patterns
      - capital/funding constraints
      - ROI/economic justification
      - surgeon/staff training
      - clinical outcomes
      - utilization
      - procurement timelines
      - future adoption
9. Return evidence indices identifying the supplied evidence supporting
   each theme.
10. A theme without sufficient supporting evidence must not be returned.
11. Keep summaries concise and descriptive.
12. Do not rank themes as "best" or "most important" unless the experts
    themselves explicitly establish that distinction.

Return JSON only.

EVIDENCE:
{evidence_block}
""".strip()

    @staticmethod
    def _theme_schema() -> dict[str, Any]:
        """
        JSON schema for grounded theme generation.
        """
        return {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme": {
                                "type": "string",
                            },
                            "summary": {
                                "type": "string",
                            },
                            "evidence_indices": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                },
                            },
                        },
                        "required": [
                            "theme",
                            "summary",
                            "evidence_indices",
                        ],
                    },
                }
            },
            "required": ["themes"],
        }

    def _build_theme_objects(
        self,
        raw_themes: list[dict[str, Any]],
        evidence: list[Evidence],
    ) -> list[Theme]:
        """
        Convert model-generated theme descriptions into grounded Theme
        models using application-controlled source metadata.
        """
        themes: list[Theme] = []
        seen_names: set[str] = set()

        for raw_theme in raw_themes:
            name = self._clean_text(
                raw_theme.get("theme")
            )

            summary = self._clean_text(
                raw_theme.get("summary")
            )

            if not name or not summary:
                continue

            normalized_name = name.casefold()

            if normalized_name in seen_names:
                continue

            indices = self._safe_indices(
                raw_theme.get("evidence_indices"),
                len(evidence),
            )

            supporting_evidence = [
                evidence[index - 1]
                for index in indices
            ]

            expert_set = {
                item.expert_name
                for item in supporting_evidence
                if item.expert_name
            }

            if len(expert_set) < self.config.minimum_experts:
                continue

            theme_evidence = self._build_theme_evidence(
                supporting_evidence
            )

            if len(theme_evidence) < self.config.minimum_evidence:
                continue

            seen_names.add(normalized_name)

            themes.append(
                Theme(
                    theme_id=self._theme_id(
                        name,
                        supporting_evidence,
                    ),
                    theme=name,
                    summary=summary,
                    experts=sorted(expert_set),
                    evidence=theme_evidence,
                )
            )

            if len(themes) >= self.config.max_themes:
                break

        return themes

    def _build_theme_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[ThemeEvidence]:
        """
        Convert verified Evidence into ThemeEvidence.
        """
        result: list[ThemeEvidence] = []

        seen: set[str] = set()

        for item in evidence:
            evidence_key = (
                f"{item.document_id}:"
                f"{item.start_timestamp}:"
                f"{item.quote}"
            )

            if evidence_key in seen:
                continue

            seen.add(evidence_key)

            citation = self.citation_builder.build(
                item
            )

            result.append(
                ThemeEvidence(
                    evidence_id=item.evidence_id,
                    expert_name=item.expert_name,
                    market=item.market,
                    quote=item.quote,
                    start_timestamp=item.start_timestamp,
                    end_timestamp=item.end_timestamp,
                    citation=citation,
                )
            )

        return result

    def _verify_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """
        Verify retrieved evidence against its own source text.

        Retrieved evidence is source-derived, but verification keeps the
        evidence contract explicit and makes future ingestion changes safer.
        """
        verified: list[Evidence] = []

        for item in evidence:
            result = self.quote_verifier.verify(
                quote=item.quote,
                source_text=item.source_text,
            )

            if not result.verified:
                logger.warning(
                    "Theme evidence failed verification: %s",
                    item.evidence_id,
                )
                continue

            verified.append(
                item.model_copy(
                    update={
                        "status": result.status,
                        "verification_message": (
                            result.message
                        ),
                    }
                )
            )

        return verified

    def _results_to_evidence(
        self,
        results: list,
    ) -> list[Evidence]:
        """
        Convert retrieval results to Evidence objects.
        """
        evidence: list[Evidence] = []

        for result in results:
            chunk = result.chunk

            if not chunk.text.strip():
                continue

            evidence.append(
                Evidence(
                    evidence_id=(
                        f"evidence:{chunk.chunk_id}"
                    ),
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    expert_name=chunk.expert_name,
                    market=chunk.market,
                    speaker=chunk.speaker,
                    start_timestamp=chunk.start_timestamp,
                    end_timestamp=chunk.end_timestamp,
                    quote=chunk.text,
                    source_text=chunk.text,
                )
            )

        return evidence

    @staticmethod
    def _serialize_evidence(
        evidence: list[Evidence],
    ) -> str:
        """
        Serialize evidence for the grounded LLM prompt.
        """
        blocks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[EVIDENCE {index}]",
                        f"Document ID: {item.document_id}",
                        f"Source file: {item.source_file}",
                        f"Expert: {item.expert_name}",
                        f"Market: {item.market}",
                        f"Timestamp: {item.start_timestamp}",
                        (
                            f"End timestamp: {item.end_timestamp}"
                            if item.end_timestamp
                            else "End timestamp: unavailable"
                        ),
                        f"Text: {item.source_text}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _normalize_experts(
        expert_names: Iterable[str] | None,
    ) -> set[str]:
        """
        Normalize optional expert names.
        """
        if expert_names is None:
            return set()

        return {
            name.strip()
            for name in expert_names
            if isinstance(name, str) and name.strip()
        }

    @staticmethod
    def _safe_indices(
        values: Any,
        evidence_count: int,
    ) -> list[int]:
        """
        Validate evidence indices returned by the LLM.
        """
        if not isinstance(values, list):
            return []

        valid: list[int] = []
        seen: set[int] = set()

        for value in values:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue

            if index < 1 or index > evidence_count:
                continue

            if index in seen:
                continue

            seen.add(index)
            valid.append(index)

        return valid

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        """
        Safely normalize generated theme text.
        """
        if not isinstance(value, str):
            return ""

        return " ".join(
            value.split()
        ).strip()

    @staticmethod
    def _theme_id(
        name: str,
        evidence: list[Evidence],
    ) -> str:
        """
        Build a deterministic theme identifier.
        """
        experts = sorted(
            {
                item.expert_name
                for item in evidence
                if item.expert_name
            }
        )

        raw = "|".join(
            [
                name.casefold(),
                *experts,
            ]
        )

        # Avoid Python's randomized hash() for persistent IDs.
        import hashlib

        digest = hashlib.sha1(
            raw.encode("utf-8")
        ).hexdigest()[:12]

        return f"theme:{digest}"


def get_theme_analyzer(
    retriever: Retriever,
    reranker: RetrievalReranker,
    *,
    llm_service: GeminiLLMService | None = None,
    quote_verifier: QuoteVerifier | None = None,
    citation_builder: CitationBuilder | None = None,
    settings: Settings | None = None,
) -> ThemeAnalyzer:
    """
    Construct a ThemeAnalyzer with application defaults.
    """
    return ThemeAnalyzer(
        retriever=retriever,
        reranker=reranker,
        llm_service=llm_service,
        quote_verifier=quote_verifier,
        citation_builder=citation_builder,
        settings=settings,
    )


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "ThemeAnalysisConfig",
    "ThemeAnalyzer",
    "get_theme_analyzer",
]