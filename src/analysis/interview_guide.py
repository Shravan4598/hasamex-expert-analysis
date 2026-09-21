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

    The analyzer follows this pipeline:

        Question
            ↓
        Retrieval
            ↓
        Reranking
            ↓
        Evidence conversion
            ↓
        Quote verification
            ↓
        Grounded LLM synthesis
            ↓
        Citation construction

    The LLM is never given responsibility for source discovery or
    timestamp generation.
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

    def analyze_question(
        self,
        question: InterviewQuestion,
        *,
        filters: RetrievalFilters | None = None,
    ) -> GroundedAnswer:
        """
        Analyze one interview question.

        Args:
            question: Interview-guide question.
            filters: Optional source filters. These are useful for
                expert-specific analysis.

        Returns:
            GroundedAnswer containing answer, evidence, and citations.
        """
        try:
            retrieval_results = self.retriever.retrieve(
                query=question.question,
                filters=filters,
                top_k=self.config.retrieval_top_k,
            )

            reranked_results = self.reranker.rerank(
                query=question.question,
                results=retrieval_results,
                top_k=self.config.rerank_top_k,
            )

            evidence = self._results_to_evidence(reranked_results)

            evidence = self._select_sufficient_evidence(evidence)

            if len(evidence) < self.config.minimum_evidence:
                return GroundedAnswer(
                    question=question.question,
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
                    refusal_reason=(
                        "No sufficiently relevant transcript evidence "
                        "was retrieved."
                    ),
                )

            grounded_answer = (
                self.llm_service.generate_grounded_answer(
                    question=question.question,
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

            return self._finalize_answer(
                grounded_answer=grounded_answer,
                evidence=verified_evidence,
                citations=citations,
            )

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Interview question analysis failed: %s",
                question.question_id,
            )
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def analyze_question_by_expert(
        self,
        question: InterviewQuestion,
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
        question: InterviewQuestion,
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
        """
        Run the complete interview guide independently for each expert.

        Returns:
            Mapping from expert name to six grounded answers.
        """
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
        """
        Convert retrieval results into source-grounded Evidence objects.

        Evidence metadata comes from retrieved transcript chunks, not
        from the LLM.
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

    def _select_sufficient_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """
        Select evidence while removing obvious duplicates.

        Retrieval scores are already incorporated into ranking before
        this stage. This method focuses on source quality and diversity.
        """
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
        """
        Verify each evidence item before allowing it into the final answer.

        Retrieved chunks originate from the transcript, so their text is
        already source-derived. The verifier is still applied here to
        enforce the same evidence contract used by generated quotes.
        """
        verified: list[Evidence] = []

        for item in evidence:
            result = self.quote_verifier.verify(
                quote=item.quote,
                source_text=item.source_text,
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
                        "verification_message": result.reason,
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
        """Apply application-level evidence safeguards to an LLM answer."""
        if not evidence:
            return grounded_answer.model_copy(
                update={
                    "evidence": [],
                    "citations": [],
                    "confidence": 0.0,
                    "evidence_coverage": 0.0,
                    "evidence_sufficient": False,
                    "refusal_reason": (
                        "The generated answer could not be supported "
                        "by verified transcript evidence."
                    ),
                }
            )

        evidence_coverage = self._calculate_evidence_coverage(
            evidence
        )

        evidence_sufficient = (
            grounded_answer.evidence_sufficient
            and evidence_coverage
            >= self.config.minimum_coverage
        )

        confidence = (
            grounded_answer.confidence
            if evidence_sufficient
            else 0.0
        )

        refusal_reason = (
            grounded_answer.refusal_reason
            if not evidence_sufficient
            else None
        )

        if not evidence_sufficient:
            answer = (
                "The available transcript evidence is not sufficient "
                "to provide a reliably grounded answer."
            )
        else:
            answer = grounded_answer.answer

        return grounded_answer.model_copy(
            update={
                "answer": answer,
                "evidence": evidence,
                "citations": citations,
                "confidence": confidence,
                "evidence_coverage": evidence_coverage,
                "evidence_sufficient": evidence_sufficient,
                "refusal_reason": refusal_reason,
            }
        )

    @staticmethod
    def _calculate_evidence_coverage(
        evidence: list[Evidence],
    ) -> float:
        """Calculate the fraction of evidence items that passed verification."""
        if not evidence:
            return 0.0

        verified_count = sum(
            1
            for item in evidence
            if item.status.value == "verified"
        )

        return verified_count / len(evidence)


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