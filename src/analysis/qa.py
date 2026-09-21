"""
Cross-transcript question answering for the Hasamex Expert Analysis app.

This module provides free-form Q&A over the expert transcripts.

Architecture:

    User Question
         |
         v
    Vector Retrieval
         |
         v
    Metadata-aware Reranking
         |
         v
    Evidence Selection
         |
         v
    Quote Verification
         |
         v
    Grounded LLM Synthesis
         |
         v
    Citation Construction
         |
         v
    GroundedAnswer

Important safety properties:

    - The LLM receives only retrieved transcript evidence.
    - Empty/insufficient retrieval results cause a safe refusal.
    - Expert names and timestamps come from source metadata.
    - Exact quote candidates are verified against source text.
    - The LLM cannot create citation metadata.
    - The answer does not use external knowledge.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from exception import SensorException
from logger import logging
from src.analysis.llm import GeminiLLMService, get_llm_service
from src.config import Settings, get_settings
from src.evidence.citation import CitationBuilder
from src.evidence.quote_verifier import QuoteVerifier
from src.models import (
    EvidenceStatus,
    GroundedAnswer,
    SearchQuery,
)
from src.retrieval.reranker import RetrievalReranker
from src.retrieval.retriever import RetrievalFilters, Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAConfig:
    """
    Runtime configuration for cross-transcript Q&A.
    """

    retrieval_top_k: int
    rerank_top_k: int
    minimum_evidence: int
    minimum_evidence_coverage: float


class TranscriptQA:
    """
    Answer user questions using source-grounded transcript evidence.

    This class is intentionally separate from the Streamlit UI so that
    the Q&A pipeline can be tested independently.
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
        Initialize the transcript Q&A service.
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

            self.config = QAConfig(
                retrieval_top_k=self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
                minimum_evidence=1,
                minimum_evidence_coverage=(
                    self.settings.min_evidence_coverage
                ),
            )

        except Exception as error:
            logger.exception(
                "Failed to initialize TranscriptQA."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def ask(
        self,
        question: str,
        *,
        expert_name: str | None = None,
        market: str | None = None,
        source_file: str | None = None,
    ) -> GroundedAnswer:
        """
        Answer a free-form question across the transcript corpus.

        Args:
            question: User's natural-language question.
            expert_name: Optional expert filter.
            market: Optional market filter.
            source_file: Optional transcript file filter.

        Returns:
            GroundedAnswer containing answer, evidence, and citations.
        """
        try:
            search_query = self._build_search_query(
                question=question,
                expert_name=expert_name,
                market=market,
                source_file=source_file,
            )

            if not search_query.query:
                return self._insufficient_evidence_answer(
                    question=question,
                    expert_name=expert_name,
                    market=market,
                    reason="The question cannot be empty.",
                )

            filters = self._build_filters(
                expert_name=expert_name,
                market=market,
                source_file=source_file,
            )

            retrieval_results = self.retriever.retrieve(
                query=search_query.query,
                filters=filters,
                top_k=self.config.retrieval_top_k,
            )

            reranked_results = self.reranker.rerank(
                query=search_query.query,
                results=retrieval_results,
                top_k=self.config.rerank_top_k,
            )

            evidence = self._retrieval_to_evidence(
                reranked_results
            )

            evidence = self._verify_evidence(
                evidence
            )

            if len(evidence) < self.config.minimum_evidence:
                return self._insufficient_evidence_answer(
                    question=question,
                    expert_name=expert_name,
                    market=market,
                    reason=(
                        "No sufficiently relevant and verified "
                        "transcript evidence was found."
                    ),
                )

            grounded_answer = (
                self.llm_service.generate_grounded_answer(
                    question=question,
                    evidence=evidence,
                    expert_name=expert_name,
                    market=market,
                )
            )

            citations = self.citation_builder.build_many(
                grounded_answer.evidence
            )

            return self._finalize_answer(
                grounded_answer=grounded_answer,
                citations=citations,
            )

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Cross-transcript Q&A failed."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def ask_query(
        self,
        search_query: SearchQuery,
    ) -> GroundedAnswer:
        """
        Answer a SearchQuery model directly.
        """
        if not isinstance(search_query, SearchQuery):
            raise TypeError(
                "search_query must be a SearchQuery instance."
            )

        return self.ask(
            question=search_query.query,
            expert_name=search_query.expert_name,
            market=search_query.market,
            source_file=search_query.source_file,
        )

    def ask_many(
        self,
        questions: Iterable[str],
    ) -> list[GroundedAnswer]:
        """
        Answer multiple questions independently.

        Each question gets its own retrieval and evidence pipeline so
        evidence from one question cannot leak into another.
        """
        answers: list[GroundedAnswer] = []

        for question in questions:
            if not isinstance(question, str):
                continue

            answers.append(
                self.ask(question)
            )

        return answers

    def ask_about_expert(
        self,
        question: str,
        expert_name: str,
    ) -> GroundedAnswer:
        """
        Ask a question restricted to one expert.
        """
        return self.ask(
            question=question,
            expert_name=expert_name,
        )

    def ask_about_market(
        self,
        question: str,
        market: str,
    ) -> GroundedAnswer:
        """
        Ask a question restricted to one market.
        """
        return self.ask(
            question=question,
            market=market,
        )

    def ask_about_source(
        self,
        question: str,
        source_file: str,
    ) -> GroundedAnswer:
        """
        Ask a question restricted to one transcript source.
        """
        return self.ask(
            question=question,
            source_file=source_file,
        )

    def _build_search_query(
        self,
        question: str,
        expert_name: str | None,
        market: str | None,
        source_file: str | None,
    ) -> SearchQuery:
        """
        Construct the application's search query model.
        """
        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        cleaned_question = " ".join(
            question.split()
        ).strip()

        return SearchQuery(
            query=cleaned_question,
            expert_name=self._clean_optional(
                expert_name
            ),
            market=self._clean_optional(
                market
            ),
            source_file=self._clean_optional(
                source_file
            ),
        )

    @staticmethod
    def _build_filters(
        expert_name: str | None,
        market: str | None,
        source_file: str | None,
    ) -> RetrievalFilters | None:
        """
        Build retrieval metadata filters when supplied.
        """
        if not any(
            [
                expert_name,
                market,
                source_file,
            ]
        ):
            return None

        return RetrievalFilters(
            expert_name=expert_name,
            market=market,
            source_file=source_file,
        )

    def _retrieval_to_evidence(
        self,
        results: list,
    ) -> list:
        """
        Convert retrieval results into source-owned Evidence objects.
        """
        evidence = []
        seen_chunk_ids: set[str] = set()

        for result in results:
            chunk = result.chunk

            if chunk.chunk_id in seen_chunk_ids:
                continue

            if not chunk.text.strip():
                continue

            seen_chunk_ids.add(
                chunk.chunk_id
            )

            from src.models import Evidence

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

    def _verify_evidence(
        self,
        evidence: list,
    ) -> list:
        """
        Verify retrieved evidence before sending it to the LLM.

        This makes the source-grounding contract explicit.
        """
        verified = []

        for item in evidence:
            verification = self.quote_verifier.verify(
                quote=item.quote,
                source_text=item.source_text,
            )

            if not verification.verified:
                logger.warning(
                    "Q&A evidence failed verification: %s",
                    item.evidence_id,
                )
                continue

            verified.append(
                item.model_copy(
                    update={
                        "status": verification.status,
                        "verification_message":
                            verification.reason,
                    }
                )
            )

        return verified

    def _finalize_answer(
        self,
        grounded_answer: GroundedAnswer,
        citations: list,
    ) -> GroundedAnswer:
        """
        Apply final application-level grounding checks.
        """
        evidence = grounded_answer.evidence

        if not evidence:
            return grounded_answer.model_copy(
                update={
                    "answer": (
                        "Insufficient verified evidence is available "
                        "in the transcripts to answer this question."
                    ),
                    "citations": [],
                    "confidence": 0.0,
                    "evidence_coverage": 0.0,
                    "evidence_sufficient": False,
                    "refusal_reason": (
                        "No verified evidence was available."
                    ),
                }
            )

        verified_count = sum(
            1
            for item in evidence
            if item.status.value == "verified"
        )

        coverage = verified_count / len(evidence)

        sufficient = (
            grounded_answer.evidence_sufficient
            and coverage
            >= self.config.minimum_evidence_coverage
        )

        if not sufficient:
            return grounded_answer.model_copy(
                update={
                    "answer": (
                        "The available transcript evidence is not "
                        "sufficient to provide a reliably grounded answer."
                    ),
                    "citations": [],
                    "confidence": 0.0,
                    "evidence_coverage": coverage,
                    "evidence_sufficient": False,
                    "refusal_reason": (
                        grounded_answer.refusal_reason
                        or (
                            "The retrieved evidence did not meet the "
                            "minimum grounding threshold."
                        )
                    ),
                }
            )

        return grounded_answer.model_copy(
            update={
                "citations": citations,
                "evidence_coverage": coverage,
                "evidence_sufficient": True,
                "evidence_status": EvidenceStatus.VERIFIED,
            }
        )

    @staticmethod
    def _insufficient_evidence_answer(
        question: str,
        expert_name: str | None,
        market: str | None,
        reason: str,
    ) -> GroundedAnswer:
        """
        Build a safe refusal without calling the LLM.
        """
        return GroundedAnswer(
            question=question,
            expert_name=expert_name,
            market=market,
            answer=(
                "Insufficient evidence in the available transcripts "
                "to answer this question reliably."
            ),
            evidence=[],
            citations=[],
            confidence=0.0,
            evidence_coverage=0.0,
            evidence_sufficient=False,
            refusal_reason=reason,
        )

    @staticmethod
    def _clean_optional(
        value: str | None,
    ) -> str | None:
        """
        Normalize an optional filter.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(
                "Filter values must be strings or None."
            )

        cleaned = " ".join(
            value.split()
        ).strip()

        return cleaned or None


def get_transcript_qa(
    retriever: Retriever,
    reranker: RetrievalReranker,
    *,
    llm_service: GeminiLLMService | None = None,
    quote_verifier: QuoteVerifier | None = None,
    citation_builder: CitationBuilder | None = None,
    settings: Settings | None = None,
) -> TranscriptQA:
    """
    Construct a TranscriptQA service using application defaults.
    """
    return TranscriptQA(
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
    "QAConfig",
    "TranscriptQA",
    "get_transcript_qa",
]