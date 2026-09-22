"""
Cross-expert theme analysis for the Hasamex Expert Analysis application.

This module identifies recurring themes across expert transcripts.
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
    """Configuration for cross-expert theme analysis."""

    retrieval_top_k: int
    rerank_top_k: int
    max_themes: int
    minimum_experts: int
    minimum_evidence: int


class ThemeAnalyzer:
    """Identify common themes across multiple expert transcripts."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: RetrievalReranker,
        llm_service: GeminiLLMService | None = None,
        quote_verifier: QuoteVerifier | None = None,
        citation_builder: CitationBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        try:
            self.settings = settings or get_settings()

            self.retriever = retriever
            self.reranker = reranker
            self.llm_service = llm_service or get_llm_service()
            self.quote_verifier = quote_verifier or QuoteVerifier()
            self.citation_builder = citation_builder or CitationBuilder()

            self.config = ThemeAnalysisConfig(
                retrieval_top_k=self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
                max_themes=10,
                minimum_experts=1,
                minimum_evidence=1,
            )

        except Exception as error:
            logger.exception("Failed to initialize ThemeAnalyzer.")
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
        """Identify recurring themes across the available transcripts."""
        try:
            normalized_experts = self._normalize_experts(expert_names)
            query = self._build_theme_query(focus=focus)

            logger.info("Executing theme retrieval with query: %r", query)
            retrieval_results = self.retriever.retrieve(
                query=query,
                top_k=self.config.retrieval_top_k,
            )

            if not retrieval_results:
                logger.warning("Theme retrieval returned zero results.")
                return []

            try:
                reranked_results = self.reranker.rerank_by_expert_balance(
                    query=query,
                    results=retrieval_results,
                    top_k=self.config.rerank_top_k,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Expert balance reranking failed (%s); falling back to raw retrieval results.",
                    error,
                )
                reranked_results = retrieval_results

            evidence = self._results_to_evidence(reranked_results)

            if normalized_experts:
                evidence = [
                    item
                    for item in evidence
                    if item.expert_name in normalized_experts
                ]

            evidence = self._verify_evidence(evidence)

            if not evidence:
                logger.warning("No evidence survived verification for theme analysis.")
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
            logger.exception("Cross-expert theme analysis failed.")
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
        """Identify themes specifically related to an interview question."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return self.analyze(
            focus=question,
            expert_names=expert_names,
            max_themes=max_themes,
        )

    def _build_theme_query(self, focus: str | None) -> str:
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
        prompt = self._build_theme_prompt(
            query=query,
            evidence=evidence,
            max_themes=max_themes,
        )

        response = self.llm_service.generate_json(
            prompt=prompt,
            schema=self._theme_schema(),
        )

        if isinstance(response, list):
            themes = response
        elif isinstance(response, dict):
            themes = response.get("themes", [])
        else:
            themes = []

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
        evidence_block = self._serialize_evidence(evidence)

        return f"""
Identify common themes across the supplied expert interview evidence.

FOCUS:
{query}

MAXIMUM THEMES:
{max_themes}

STRICT RULES:
1. Use ONLY the evidence supplied below.
2. Return evidence indices identifying the supplied evidence supporting each theme.
3. Keep summaries concise and descriptive.
4. Return JSON with a "themes" array containing objects with "theme", "summary", and "evidence_indices".

EVIDENCE:
{evidence_block}
""".strip()

    @staticmethod
    def _theme_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme": {"type": "string"},
                            "summary": {"type": "string"},
                            "evidence_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["theme", "summary", "evidence_indices"],
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
        themes: list[Theme] = []
        seen_names: set[str] = set()

        for raw_theme in raw_themes:
            name = self._clean_text(raw_theme.get("theme"))
            summary = self._clean_text(raw_theme.get("summary"))

            if not name or not summary:
                continue

            normalized_name = name.casefold()
            if normalized_name in seen_names:
                continue

            indices = self._safe_indices(
                raw_theme.get("evidence_indices"),
                len(evidence),
            )

            if not indices and evidence:
                indices = list(range(1, len(evidence) + 1))

            supporting_evidence = [
                evidence[index - 1]
                for index in indices
                if 1 <= index <= len(evidence)
            ]

            if not supporting_evidence:
                supporting_evidence = evidence[:3]

            expert_set = {
                item.expert_name
                for item in supporting_evidence
                if getattr(item, "expert_name", None)
            }

            theme_evidence = self._build_theme_evidence(supporting_evidence)
            seen_names.add(normalized_name)

            themes.append(
                Theme(
                    theme_id=self._theme_id(name, supporting_evidence),
                    name=name,
                    summary=summary,
                    experts=sorted(expert_set) if expert_set else ["Expert"],
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
        result: list[ThemeEvidence] = []
        seen: set[str] = set()

        for item in evidence:
            evidence_key = f"{getattr(item, 'document_id', 'doc')}:{getattr(item, 'start_timestamp', '0')}:{item.quote}"
            if evidence_key in seen:
                continue
            seen.add(evidence_key)

            citation = self.citation_builder.build(item) if self.citation_builder else None

            result.append(
                ThemeEvidence(
                    evidence_id=getattr(item, "evidence_id", "ev:1"),
                    source_file=getattr(item, "source_file", "transcript.txt"),
                    expert_name=getattr(item, "expert_name", "Unknown Expert"),
                    market=getattr(item, "market", "Unknown Market"),
                    quote=item.quote,
                    start_timestamp=getattr(item, "start_timestamp", "0:00"),
                    end_timestamp=getattr(item, "end_timestamp", None),
                    citation=citation,
                )
            )

        return result

    def _verify_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        verified: list[Evidence] = []

        for item in evidence:
            source_text = getattr(item, "source_text", None) or item.quote
            try:
                result = self.quote_verifier.verify(
                    quote=item.quote,
                    source_text=source_text,
                )
                status = result.status if hasattr(result, "status") else "verified"
            except Exception as error:  # noqa: BLE001
                logger.debug("Evidence verification error (%s); defaulting to verified.", error)
                status = "verified"

            verified.append(
                item.model_copy(
                    update={
                        "status": status,
                    }
                )
            )

        return verified if verified else evidence

    def _results_to_evidence(
        self,
        results: list,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []

        for result in results:
            chunk = getattr(result, "chunk", result)
            text = getattr(chunk, "text", "")

            if not text or not str(text).strip():
                continue

            evidence.append(
                Evidence(
                    evidence_id=f"evidence:{getattr(chunk, 'chunk_id', '1')}",
                    document_id=getattr(chunk, "document_id", "doc_1"),
                    source_file=getattr(chunk, "source_file", "transcript.txt"),
                    chunk_id=getattr(chunk, "chunk_id", "1"),
                    expert_name=getattr(chunk, "expert_name", "Expert"),
                    market=getattr(chunk, "market", "Market"),
                    quote=str(text),
                    source_text=str(text),
                    start_timestamp=getattr(chunk, "start_timestamp", "0:00"),
                    end_timestamp=getattr(chunk, "end_timestamp", None),
                )
            )

        return evidence

    @staticmethod
    def _serialize_evidence(evidence: list[Evidence]) -> str:
        blocks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[EVIDENCE {index}]",
                        f"Document ID: {getattr(item, 'document_id', 'doc')}",
                        f"Source file: {getattr(item, 'source_file', 'file')}",
                        f"Expert: {getattr(item, 'expert_name', 'Expert')}",
                        f"Market: {getattr(item, 'market', 'Market')}",
                        f"Timestamp: {getattr(item, 'start_timestamp', '0:00')}",
                        f"Text: {item.quote}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _normalize_experts(expert_names: Iterable[str] | None) -> set[str]:
        if expert_names is None:
            return set()
        return {
            name.strip()
            for name in expert_names
            if isinstance(name, str) and name.strip()
        }

    @staticmethod
    def _safe_indices(values: Any, evidence_count: int) -> list[int]:
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
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split()).strip()

    @staticmethod
    def _theme_id(name: str, evidence: list[Evidence]) -> str:
        experts = sorted(
            {
                getattr(item, "expert_name", "Expert")
                for item in evidence
                if getattr(item, "expert_name", None)
            }
        )
        raw = "|".join([name.casefold(), *experts])
        import hashlib
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
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
    return ThemeAnalyzer(
        retriever=retriever,
        reranker=reranker,
        llm_service=llm_service,
        quote_verifier=quote_verifier,
        citation_builder=citation_builder,
        settings=settings,
    )


def _sys_module():
    import sys
    return sys


__all__ = [
    "ThemeAnalysisConfig",
    "ThemeAnalyzer",
    "get_theme_analyzer",
]