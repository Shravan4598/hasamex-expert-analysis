"""
Interview-guide analysis for the Hasamex Expert Analysis application.

This module manages the fixed interview questions supplied in the
Hasamex case study and coordinates retrieval, reranking, evidence
selection, quote verification, citation construction, and grounded
LLM analysis.

The module deliberately does not allow the LLM to decide which source
documents exist. Source selection is controlled by the application.
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
    Citation,
    Evidence,
    GroundedAnswer,
    InterviewQuestion,
    RetrievalResult,
)
from src.retrieval.reranker import RetrievalReranker
from src.retrieval.retriever import RetrievalFilters, Retriever

logger = logging.getLogger(__name__)


INTERVIEW_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion(
        question_id="Q1",
        question=(
            "How would you describe current adoption of robotic "
            "surgery in your market?"
        ),
        topic="Current adoption",
    ),
    InterviewQuestion(
        question_id="Q2",
        question="What are the main barriers to adoption?",
        topic="Adoption barriers",
    ),
    InterviewQuestion(
        question_id="Q3",
        question=(
            "How important are hospital budgets and ROI in "
            "purchasing decisions?"
        ),
        topic="Budget and ROI",
    ),
    InterviewQuestion(
        question_id="Q4",
        question=(
            "How important are surgeon training and clinical outcomes?"
        ),
        topic="Training and clinical outcomes",
    ),
    InterviewQuestion(
        question_id="Q5",
        question=(
            "What adoption trend do you expect over the next 3–5 years?"
        ),
        topic="Future adoption",
    ),
    InterviewQuestion(
        question_id="Q6",
        question=(
            "What is the typical hospital decision-making timeline "
            "for purchasing a new robotic system?"
        ),
        topic="Purchase timeline",
    ),
)


@dataclass(frozen=True)
class InterviewAnalysisConfig:
    """Configuration for interview-guide analysis."""

    retrieval_top_k: int
    rerank_top_k: int
    minimum_evidence: int
    minimum_coverage: float


class InterviewGuideAnalyzer:
    """
    Analyze the fixed interview guide against transcript evidence.
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
        """Initialize the interview-guide analyzer."""
        try:
            self.settings = settings or get_settings()

            self.retriever = retriever
            self.reranker = reranker
            self.llm_service = llm_service or get_llm_service()
            self.quote_verifier = quote_verifier or QuoteVerifier()
            self.citation_builder = (
                citation_builder or CitationBuilder()
            )

            self.config = InterviewAnalysisConfig(
                retrieval_top_k=self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
                minimum_evidence=1,
                minimum_coverage=self.settings.min_evidence_coverage,
            )

        except Exception as error:
            logger.exception(
                "Failed to initialize InterviewGuideAnalyzer."
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def get_questions(self) -> list[InterviewQuestion]:
        """Return the six official interview-guide questions."""
        return list(INTERVIEW_QUESTIONS)

    def get_question(
        self,
        question_id: str,
    ) -> InterviewQuestion:
        """Retrieve one interview question by ID."""
        normalized_id = question_id.strip().upper()

        for question in INTERVIEW_QUESTIONS:
            if question.question_id == normalized_id:
                return question

        raise ValueError(
            f"Unknown interview question ID: {question_id}"
        )

    def _resolve_question(
        self,
        question: InterviewQuestion | str,
    ) -> InterviewQuestion:
        """Robustly resolve question whether passed as object, ID, or text."""
        if isinstance(question, InterviewQuestion):
            return question
        
        q_str = str(question).strip()
        # Try matching by ID first
        try:
            return self.get_question(q_str)
        except ValueError:
            pass

        # Try matching by text
        for q in INTERVIEW_QUESTIONS:
            if q.question.lower() == q_str.lower():
                return q

        # Fallback default if not found
        return InterviewQuestion(
            question_id="CUSTOM",
            question=q_str,
            topic="Custom query",
        )

    def analyze_question(
        self,
        question: InterviewQuestion | str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> GroundedAnswer:
        """
        Analyze one interview question. Accepts either an InterviewQuestion object or a string.
        """
        resolved_question = self._resolve_question(question)

        try:
            retrieval_results = self.retriever.retrieve(
                query=resolved_question.question,
                filters=filters,
                top_k=self.config.retrieval_top_k,
            )

            reranked_results = self.reranker.rerank(
                query=resolved_question.question,
                results=retrieval_results,
                top_k=self.config.rerank_top_k,
            )

            evidence = self._results_to_evidence(reranked_results)

            evidence = self._select_sufficient_evidence(evidence)

            if len(evidence) < self.config.minimum_evidence:
                return GroundedAnswer(
                    question=resolved_question.question,
                    question_id=resolved_question.question_id,
                    expert_name=(
                        filters.expert_name
                        if filters
                        else None
                    ),
                    market=(
                        filters.market
                        if filters
                        else None
                    ),
                    answer=(
                        "Insufficient evidence in the available "
                        "transcript sources to answer this question."
                    ),
                    evidence=[],
                    citations=[],
                    confidence=0.0,
                    evidence_coverage=0.0,
                    evidence_sufficient=False,
                    reasoning=(
                        "No sufficiently relevant transcript evidence "
                        "was retrieved."
                    ),
                )

            grounded_answer = (
                self.llm_service.generate_grounded_answer(
                    question=resolved_question.question,
                    evidence=evidence,
                    expert_name=(
                        filters.expert_name
                        if filters
                        else None
                    ),
                    market=(
                        filters.market
                        if filters
                        else None
                    ),
                )
            )

            verified_evidence = self._verify_answer_evidence(
                grounded_answer.evidence
            )

            citations = self.citation_builder.build_many(
                verified_evidence
            )

            # Ensure question_id is populated on the grounded answer
            final_answer = self._finalize_answer(
                grounded_answer=grounded_answer,
                evidence=verified_evidence,
                citations=citations,
            )
            
            if not final_answer.question_id:
                final_answer.question_id = resolved_question.question_id

            return final_answer

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Interview question analysis failed: %s",
                resolved_question.question_id,
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def analyze_question_by_expert(
        self,
        question: InterviewQuestion | str,
        expert_name: str,
    ) -> GroundedAnswer:
        """Analyze one interview question for one expert."""
        filters = RetrievalFilters(
            expert_name=expert_name
        )

        return self.analyze_question(
            question=question,
            filters=filters,
        )

    def analyze_question_by_market(
        self,
        question: InterviewQuestion | str,
        market: str,
    ) -> GroundedAnswer:
        """Analyze one interview question for one market."""
        filters = RetrievalFilters(
            market=market
        )

        return self.analyze_question(
            question=question,
            filters=filters,
        )

    def analyze_all_questions(
        self,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[GroundedAnswer]:
        """Analyze all six interview-guide questions."""
        results: list[GroundedAnswer] = []

        for question in INTERVIEW_QUESTIONS:
            results.append(
                self.analyze_question(
                    question=question,
                    filters=filters,
                )
            )

        return results

    def analyze_all_for_expert(
        self,
        expert_name: str,
    ) -> list[GroundedAnswer]:
        """Answer all interview-guide questions for one expert."""
        return self.analyze_all_questions(
            filters=RetrievalFilters(
                expert_name=expert_name
            )
        )

    def analyze_all_for_market(
        self,
        market: str,
    ) -> list[GroundedAnswer]:
        """Answer all interview-guide questions for one market."""
        return self.analyze_all_questions(
            filters=RetrievalFilters(
                market=market
            )
        )

    def analyze_all_experts(
        self,
        expert_names: Iterable[str],
    ) -> dict[str, list[GroundedAnswer]]:
        """Run the complete interview guide independently for each expert."""
        results: dict[str, list[GroundedAnswer]] = {}

        for expert_name in expert_names:
            normalized_name = expert_name.strip()

            if not normalized_name:
                continue

            results[normalized_name] = (
                self.analyze_all_for_expert(
                    normalized_name
                )
            )

        return results

    def _results_to_evidence(
        self,
        results: list[RetrievalResult],
    ) -> list[Evidence]:
        evidence: list[Evidence] = []

        for result in results:
            chunk = result.chunk

            if not chunk or not chunk.text.strip():
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
                    quote=chunk.text,
                    start_timestamp=chunk.start_timestamp,
                    end_timestamp=chunk.end_timestamp,
                )
            )

        return evidence

    def _select_sufficient_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        if not evidence:
            return []

        selected: list[Evidence] = []
        seen_chunks: set[str] = set()

        for item in evidence:
            if item.chunk_id in seen_chunks:
                continue

            if not item.quote.strip():
                continue

            seen_chunks.add(item.chunk_id)
            selected.append(item)

        return selected

    def _verify_answer_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        verified: list[Evidence] = []

        for item in evidence:
            result = self.quote_verifier.verify(
                quote=item.quote,
                source_text=getattr(item, "source_text", item.quote),
            )

            if not result.verified:
                logger.warning(
                    "Evidence failed quote verification: %s",
                    item.evidence_id,
                )
                continue

            verified.append(
                item.model_copy(
                    update={
                        "status": result.status,
                    }
                )
            )

        return verified

    def _finalize_answer(
        self,
        grounded_answer: GroundedAnswer,
        evidence: list[Evidence],
        citations: list[Citation],
    ) -> GroundedAnswer:
        if not evidence:
            return grounded_answer.model_copy(
                update={
                    "evidence": [],
                    "citations": [],
                    "confidence": 0.0,
                    "reasoning": (
                        "The generated answer could not be supported "
                        "by verified transcript evidence."
                    ),
                }
            )

        return grounded_answer.model_copy(
            update={
                "evidence": evidence,
                "citations": citations,
            }
        )


def get_interview_questions() -> list[InterviewQuestion]:
    """Return the complete official interview guide."""
    return list(INTERVIEW_QUESTIONS)


def get_interview_question(
    question_id: str,
) -> InterviewQuestion:
    """Return one official interview question."""
    normalized_id = question_id.strip().upper()

    for question in INTERVIEW_QUESTIONS:
        if question.question_id == normalized_id:
            return question

    raise ValueError(
        f"Unknown interview question ID: {question_id}"
    )


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "INTERVIEW_QUESTIONS",
    "InterviewAnalysisConfig",
    "InterviewGuideAnalyzer",
    "get_interview_question",
    "get_interview_questions",
]