"""
End-to-end evaluation runner for the Hasamex Expert Analysis application.

The evaluator runs the configured cases from ``evaluation/questions.json``
against the same retrieval and analysis pipeline used by the application.

Evaluation goals:

- Verify that cases execute successfully.
- Verify that grounded evidence is returned.
- Verify citation availability.
- Verify evidence sufficiency.
- Verify expert attribution when expected.
- Verify market attribution when expected.
- Verify timestamp traceability when expected.
- Verify expected keywords.
- Verify expected quote fragments.
- Support intentional insufficient-evidence cases.
- Produce a machine-readable JSON report.

The evaluator does NOT claim general model accuracy.

A passing case means that the configured expectations for that specific
evaluation case were satisfied during that run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis.qa import TranscriptQA
from src.config import get_settings
from src.models import EvaluationCase, EvaluationResult, GroundedAnswer
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FAISSVectorStore


LOGGER = logging.getLogger(__name__)


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "questions.json"


@dataclass
class EvaluationSummary:
    """Aggregate results for an evaluation run."""

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    execution_successes: int = 0
    evidence_cases: int = 0
    citation_cases: int = 0
    sufficient_evidence_cases: int = 0
    expert_matches: int = 0
    market_matches: int = 0
    timestamp_matches: int = 0
    keyword_matches: int = 0
    quote_matches: int = 0

    @property
    def pass_rate(self) -> float:
        """Return the percentage of passed cases."""
        if self.total_cases == 0:
            return 0.0

        return self.passed_cases / self.total_cases

    @property
    def execution_rate(self) -> float:
        """Return the percentage of cases that executed successfully."""
        if self.total_cases == 0:
            return 0.0

        return self.execution_successes / self.total_cases

    @property
    def evidence_rate(self) -> float:
        """Return the percentage of cases returning evidence."""
        if self.total_cases == 0:
            return 0.0

        return self.evidence_cases / self.total_cases

    @property
    def citation_rate(self) -> float:
        """Return the percentage of cases returning citations."""
        if self.total_cases == 0:
            return 0.0

        return self.citation_cases / self.total_cases

    @property
    def grounding_rate(self) -> float:
        """Return the percentage of cases with sufficient evidence."""
        if self.total_cases == 0:
            return 0.0

        return self.sufficient_evidence_cases / self.total_cases


class EvaluationRunner:
    """
    Execute evaluation cases against the production analysis pipeline.

    The runner intentionally does not implement a separate retrieval or
    generation mechanism. This prevents the evaluation from testing a
    different system than the one exposed to users.
    """

    def __init__(
        self,
        qa_engine: TranscriptQA,
    ) -> None:
        self.qa_engine = qa_engine

    def evaluate_case(
        self,
        case: EvaluationCase,
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate one configured case.

        Parameters
        ----------
        case:
            Evaluation case from questions.json.

        verbose:
            Whether detailed diagnostic information should be logged.

        Returns
        -------
        EvaluationResult
            Structured result for the case.
        """
        LOGGER.info(
            "Evaluating case %s: %s",
            case.case_id,
            case.question,
        )

        try:
            answer = self._execute_case(case)

            result = self._evaluate_answer(
                case=case,
                answer=answer,
            )

            if verbose:
                self._log_case_result(
                    case=case,
                    answer=answer,
                    result=result,
                )

            return result

        except Exception as exc:
            LOGGER.exception(
                "Evaluation case %s failed during execution.",
                case.case_id,
            )

            return EvaluationResult(
                case_id=case.case_id,
                passed=False,
                execution_success=False,
                evidence_present=False,
                citations_present=False,
                evidence_sufficient=False,
                expert_match=False,
                market_match=False,
                timestamp_match=False,
                keyword_match=False,
                quote_match=False,
                evidence_coverage=0.0,
                failure_reasons=[
                    f"Execution error: {exc}",
                ],
            )

    def _execute_case(
        self,
        case: EvaluationCase,
    ) -> GroundedAnswer:
        """Execute an evaluation case through TranscriptQA."""
        return self.qa_engine.ask_query(
            question=case.question,
            expert_name=case.expert_name,
            market=case.market,
        )

    def _evaluate_answer(
        self,
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> EvaluationResult:
        """Evaluate the returned grounded answer against case expectations."""
        evidence_present = bool(answer.evidence)

        citations_present = bool(answer.citations)

        evidence_sufficient = bool(
            answer.evidence_sufficient
        )

        evidence_coverage = float(
            answer.evidence_coverage or 0.0
        )

        expert_match = self._check_expert(
            case=case,
            answer=answer,
        )

        market_match = self._check_market(
            case=case,
            answer=answer,
        )

        timestamp_match = self._check_timestamp(
            case=case,
            answer=answer,
        )

        keyword_match = self._check_keywords(
            case=case,
            answer=answer,
        )

        quote_match = self._check_quote_fragments(
            case=case,
            answer=answer,
        )

        failure_reasons: list[str] = []

        if not evidence_present:
            failure_reasons.append(
                "No evidence was returned."
            )

        if not citations_present and evidence_present:
            failure_reasons.append(
                "No citations were returned."
            )

        if not evidence_sufficient and self._requires_evidence(case):
            failure_reasons.append(
                "The answer did not establish sufficient evidence."
            )

        if case.expected_expert and not expert_match:
            failure_reasons.append(
                f"Expected expert '{case.expected_expert}' "
                "was not found in the answer evidence."
            )

        if case.expected_market and not market_match:
            failure_reasons.append(
                f"Expected market '{case.expected_market}' "
                "was not found in the answer evidence."
            )

        if case.expected_timestamp and not timestamp_match:
            failure_reasons.append(
                f"Expected timestamp '{case.expected_timestamp}' "
                "was not found in the answer evidence or citations."
            )

        if case.expected_keywords and not keyword_match:
            failure_reasons.append(
                "Expected keywords were not sufficiently represented "
                "in the grounded answer/evidence."
            )

        if case.expected_quote_fragments and not quote_match:
            failure_reasons.append(
                "Expected quote fragments were not found in the "
                "returned evidence."
            )

        passed = self._determine_pass_status(
            case=case,
            evidence_present=evidence_present,
            citations_present=citations_present,
            evidence_sufficient=evidence_sufficient,
            expert_match=expert_match,
            market_match=market_match,
            timestamp_match=timestamp_match,
            keyword_match=keyword_match,
            quote_match=quote_match,
        )

        return EvaluationResult(
            case_id=case.case_id,
            passed=passed,
            execution_success=True,
            evidence_present=evidence_present,
            citations_present=citations_present,
            evidence_sufficient=evidence_sufficient,
            expert_match=expert_match,
            market_match=market_match,
            timestamp_match=timestamp_match,
            keyword_match=keyword_match,
            quote_match=quote_match,
            evidence_coverage=evidence_coverage,
            failure_reasons=failure_reasons,
        )

    @staticmethod
    def _requires_evidence(
        case: EvaluationCase,
    ) -> bool:
        """
        Determine whether a case should normally require evidence.

        Cases whose expected evidence is explicitly absent are allowed
        to pass when the application correctly refuses to answer.
        """
        return not (
            case.expected_expert is None
            and case.expected_market is None
            and case.expected_timestamp is None
            and not case.expected_keywords
            and not case.expected_quote_fragments
            and case.case_type.lower()
            in {
                "insufficient_evidence",
                "unsupported",
                "refusal",
            }
        )

    @staticmethod
    def _determine_pass_status(
        case: EvaluationCase,
        evidence_present: bool,
        citations_present: bool,
        evidence_sufficient: bool,
        expert_match: bool,
        market_match: bool,
        timestamp_match: bool,
        keyword_match: bool,
        quote_match: bool,
    ) -> bool:
        """
        Determine whether a case passes.

        For ordinary grounded cases, evidence and citations are required.

        For intentional insufficient-evidence cases, the desired behavior
        is a safe refusal rather than fabricated evidence.
        """
        case_type = case.case_type.lower().strip()

        is_insufficient_case = case_type in {
            "insufficient_evidence",
            "unsupported",
            "refusal",
        }

        if is_insufficient_case:
            return not evidence_sufficient

        if not evidence_present:
            return False

        if not citations_present:
            return False

        if not evidence_sufficient:
            return False

        if case.expected_expert and not expert_match:
            return False

        if case.expected_market and not market_match:
            return False

        if case.expected_timestamp and not timestamp_match:
            return False

        if case.expected_keywords and not keyword_match:
            return False

        if case.expected_quote_fragments and not quote_match:
            return False

        return True

    @staticmethod
    def _check_expert(
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> bool:
        """Check expected expert attribution."""
        if not case.expected_expert:
            return True

        expected = _normalize(case.expected_expert)

        if answer.expert_name:
            if _normalize(answer.expert_name) == expected:
                return True

        for evidence in answer.evidence:
            if _normalize(evidence.expert_name) == expected:
                return True

        for citation in answer.citations:
            if _normalize(citation.expert_name) == expected:
                return True

        return False

    @staticmethod
    def _check_market(
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> bool:
        """Check expected market attribution."""
        if not case.expected_market:
            return True

        expected = _normalize(case.expected_market)

        if answer.market:
            if _normalize(answer.market) == expected:
                return True

        for evidence in answer.evidence:
            if _normalize(evidence.market) == expected:
                return True

        for citation in answer.citations:
            if _normalize(citation.market) == expected:
                return True

        return False

    @staticmethod
    def _check_timestamp(
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> bool:
        """Check whether the expected timestamp is traceable."""
        if not case.expected_timestamp:
            return True

        expected = _normalize_timestamp(
            case.expected_timestamp
        )

        for evidence in answer.evidence:
            if _timestamp_matches(
                expected,
                evidence.start_timestamp,
                evidence.end_timestamp,
            ):
                return True

        for citation in answer.citations:
            if _timestamp_matches(
                expected,
                citation.start_timestamp,
                citation.end_timestamp,
            ):
                return True

        return False

    @staticmethod
    def _check_keywords(
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> bool:
        """
        Check expected keywords against grounded answer material.

        Matching is case-insensitive and operates over:

        - generated answer;
        - verified/unverified evidence quotes;
        - source text.
        """
        if not case.expected_keywords:
            return True

        searchable_parts = [
            answer.answer or "",
        ]

        for evidence in answer.evidence:
            searchable_parts.extend(
                [
                    evidence.quote or "",
                    evidence.source_text or "",
                ]
            )

        searchable_text = _normalize(
            " ".join(searchable_parts)
        )

        matched = 0

        for keyword in case.expected_keywords:
            normalized_keyword = _normalize(keyword)

            if normalized_keyword and normalized_keyword in searchable_text:
                matched += 1

        return matched == len(case.expected_keywords)

    @staticmethod
    def _check_quote_fragments(
        case: EvaluationCase,
        answer: GroundedAnswer,
    ) -> bool:
        """Check whether expected source quote fragments are present."""
        if not case.expected_quote_fragments:
            return True

        evidence_text = " ".join(
            [
                evidence.quote or ""
                for evidence in answer.evidence
            ]
        )

        evidence_text = _normalize(evidence_text)

        for fragment in case.expected_quote_fragments:
            normalized_fragment = _normalize(fragment)

            if not normalized_fragment:
                continue

            if normalized_fragment not in evidence_text:
                return False

        return True

    @staticmethod
    def _log_case_result(
        case: EvaluationCase,
        answer: GroundedAnswer,
        result: EvaluationResult,
    ) -> None:
        """Log detailed information for one evaluation case."""
        status = "PASS" if result.passed else "FAIL"

        LOGGER.info(
            "[%s] %s",
            status,
            case.case_id,
        )

        LOGGER.info(
            "Answer: %s",
            answer.answer,
        )

        LOGGER.info(
            "Evidence count: %d",
            len(answer.evidence),
        )

        LOGGER.info(
            "Citation count: %d",
            len(answer.citations),
        )

        LOGGER.info(
            "Evidence sufficient: %s",
            answer.evidence_sufficient,
        )

        LOGGER.info(
            "Evidence coverage: %.3f",
            answer.evidence_coverage,
        )

        if result.failure_reasons:
            for reason in result.failure_reasons:
                LOGGER.info(
                    "Failure reason: %s",
                    reason,
                )

    def run(
        self,
        cases: list[EvaluationCase],
        verbose: bool = False,
    ) -> tuple[list[EvaluationResult], EvaluationSummary]:
        """Run every configured evaluation case."""
        results: list[EvaluationResult] = []

        summary = EvaluationSummary(
            total_cases=len(cases),
        )

        for case in cases:
            result = self.evaluate_case(
                case=case,
                verbose=verbose,
            )

            results.append(result)
            self._update_summary(
                summary=summary,
                result=result,
            )

        return results, summary

    @staticmethod
    def _update_summary(
        summary: EvaluationSummary,
        result: EvaluationResult,
    ) -> None:
        """Update aggregate counters from one case result."""
        if result.passed:
            summary.passed_cases += 1
        else:
            summary.failed_cases += 1

        if result.execution_success:
            summary.execution_successes += 1

        if result.evidence_present:
            summary.evidence_cases += 1

        if result.citations_present:
            summary.citation_cases += 1

        if result.evidence_sufficient:
            summary.sufficient_evidence_cases += 1

        if result.expert_match:
            summary.expert_matches += 1

        if result.market_match:
            summary.market_matches += 1

        if result.timestamp_match:
            summary.timestamp_matches += 1

        if result.keyword_match:
            summary.keyword_matches += 1

        if result.quote_match:
            summary.quote_matches += 1


def load_cases(
    cases_path: Path,
) -> list[EvaluationCase]:
    """
    Load and validate evaluation cases from JSON.

    Raises
    ------
    FileNotFoundError
        If the evaluation file does not exist.

    ValueError
        If the JSON structure is invalid.
    """
    if not cases_path.exists():
        raise FileNotFoundError(
            f"Evaluation cases file not found: {cases_path}"
        )

    try:
        payload = json.loads(
            cases_path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in evaluation cases file: {cases_path}"
        ) from exc

    if isinstance(payload, dict):
        raw_cases = payload.get("cases")

        if raw_cases is None:
            raise ValueError(
                "Evaluation JSON object must contain a 'cases' list."
            )
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raise ValueError(
            "Evaluation JSON must contain either a list of cases "
            "or an object with a 'cases' list."
        )

    if not isinstance(raw_cases, list):
        raise ValueError(
            "The 'cases' field must be a list."
        )

    cases: list[EvaluationCase] = []

    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Evaluation case at index {index} is not an object."
            )

        try:
            cases.append(
                EvaluationCase.model_validate(raw_case)
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid evaluation case at index {index}: {exc}"
            ) from exc

    if not cases:
        raise ValueError(
            "No evaluation cases were found."
        )

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )

    return cases


def build_evaluation_runner() -> EvaluationRunner:
    """
    Build the same retrieval/analysis dependencies used by the application.
    """
    settings = get_settings()

    settings.ensure_directories()

    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
    )

    vector_store = FAISSVectorStore(
        storage_dir=settings.vector_store_path,
    )

    if not vector_store.exists():
        raise FileNotFoundError(
            "Vector store does not exist.\n"
            "Run the ingestion pipeline first:\n\n"
            "    python scripts/ingest.py\n"
        )

    vector_store.load()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
    )

    reranker = Reranker(
        top_k=settings.rerank_top_k,
    )

    qa_engine = TranscriptQA(
        retriever=retriever,
        reranker=reranker,
        settings=settings,
    )

    return EvaluationRunner(
        qa_engine=qa_engine,
    )


def build_output_payload(
    results: list[EvaluationResult],
    summary: EvaluationSummary,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    """Build a JSON-serializable evaluation report."""
    case_lookup = {
        case.case_id: case
        for case in cases
    }

    result_payload: list[dict[str, Any]] = []

    for result in results:
        case = case_lookup.get(result.case_id)

        result_payload.append(
            {
                "case": (
                    case.model_dump(mode="json")
                    if case
                    else None
                ),
                "result": result.model_dump(mode="json"),
            }
        )

    return {
        "evaluation": {
            "total_cases": summary.total_cases,
            "passed_cases": summary.passed_cases,
            "failed_cases": summary.failed_cases,
            "pass_rate": round(
                summary.pass_rate,
                4,
            ),
            "execution_successes": summary.execution_successes,
            "execution_rate": round(
                summary.execution_rate,
                4,
            ),
            "evidence_cases": summary.evidence_cases,
            "evidence_rate": round(
                summary.evidence_rate,
                4,
            ),
            "citation_cases": summary.citation_cases,
            "citation_rate": round(
                summary.citation_rate,
                4,
            ),
            "sufficient_evidence_cases": (
                summary.sufficient_evidence_cases
            ),
            "grounding_rate": round(
                summary.grounding_rate,
                4,
            ),
            "expert_matches": summary.expert_matches,
            "market_matches": summary.market_matches,
            "timestamp_matches": summary.timestamp_matches,
            "keyword_matches": summary.keyword_matches,
            "quote_matches": summary.quote_matches,
        },
        "results": result_payload,
    }


def save_report(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write the evaluation report as formatted JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def print_summary(
    summary: EvaluationSummary,
) -> None:
    """Print a concise human-readable evaluation summary."""
    print()
    print("=" * 64)
    print("HASAMEX EXPERT ANALYSIS - EVALUATION SUMMARY")
    print("=" * 64)

    print(
        f"Total cases          : {summary.total_cases}"
    )
    print(
        f"Passed               : {summary.passed_cases}"
    )
    print(
        f"Failed               : {summary.failed_cases}"
    )
    print(
        f"Pass rate            : {summary.pass_rate:.1%}"
    )
    print(
        f"Execution success    : {summary.execution_rate:.1%}"
    )
    print(
        f"Evidence returned    : {summary.evidence_rate:.1%}"
    )
    print(
        f"Citations returned   : {summary.citation_rate:.1%}"
    )
    print(
        f"Evidence sufficient  : {summary.grounding_rate:.1%}"
    )

    print()
    print("Configured checks")
    print("-" * 64)

    print(
        f"Expert matches       : {summary.expert_matches}"
    )
    print(
        f"Market matches       : {summary.market_matches}"
    )
    print(
        f"Timestamp matches    : {summary.timestamp_matches}"
    )
    print(
        f"Keyword matches      : {summary.keyword_matches}"
    )
    print(
        f"Quote matches        : {summary.quote_matches}"
    )

    print("=" * 64)
    print()


def print_failures(
    results: list[EvaluationResult],
) -> None:
    """Print detailed failure information."""
    failures = [
        result
        for result in results
        if not result.passed
    ]

    if not failures:
        return

    print()
    print("FAILED CASES")
    print("-" * 64)

    for result in failures:
        print(f"\n{result.case_id}")

        if not result.failure_reasons:
            print("  - No diagnostic reason was recorded.")
            continue

        for reason in result.failure_reasons:
            print(f"  - {reason}")


def configure_logging(
    verbose: bool,
) -> None:
    """Configure console logging for the evaluation runner."""
    level = (
        logging.DEBUG
        if verbose
        else logging.INFO
    )

    logging.basicConfig(
        level=level,
        format=(
            "[%(asctime)s] "
            "%(levelname)s "
            "%(name)s - "
            "%(message)s"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run grounded evaluation cases for "
            "the Hasamex Expert Analysis application."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "Path to evaluation cases JSON. "
            "Defaults to evaluation/questions.json."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path for the generated JSON evaluation report."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation diagnostics.",
    )

    return parser.parse_args()


def _normalize(value: str | None) -> str:
    """Normalize text for deterministic case-insensitive matching."""
    if not value:
        return ""

    return " ".join(
        value.casefold().split()
    )


def _normalize_timestamp(
    timestamp: str | None,
) -> str:
    """Normalize timestamp strings for comparison."""
    if not timestamp:
        return ""

    value = timestamp.strip()

    if len(value) == 4:
        return f"00:{value}"

    return value


def _timestamp_matches(
    expected: str,
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> bool:
    """Return whether an expected timestamp falls in the evidence range."""
    if not expected:
        return True

    start = _normalize_timestamp(
        start_timestamp,
    )

    end = _normalize_timestamp(
        end_timestamp,
    )

    if expected == start or expected == end:
        return True

    # For point-in-time evidence, the start timestamp is normally the
    # authoritative citation timestamp. We intentionally do not infer
    # arbitrary timestamps inside a segment.
    return False


def main() -> int:
    """Run the complete evaluation workflow."""
    args = parse_arguments()

    configure_logging(
        verbose=args.verbose,
    )

    try:
        cases = load_cases(
            cases_path=args.cases,
        )

        LOGGER.info(
            "Loaded %d evaluation cases from %s.",
            len(cases),
            args.cases,
        )

        runner = build_evaluation_runner()

        results, summary = runner.run(
            cases=cases,
            verbose=args.verbose,
        )

        print_summary(summary)

        print_failures(results)

        if args.output:
            payload = build_output_payload(
                results=results,
                summary=summary,
                cases=cases,
            )

            save_report(
                output_path=args.output,
                payload=payload,
            )

            print(
                f"Evaluation report written to: {args.output}"
            )

        # Return a non-zero exit code when a configured evaluation
        # case fails. This makes the evaluator suitable for CI/CD.
        if summary.failed_cases > 0:
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        return 130

    except Exception as exc:
        LOGGER.exception(
            "Evaluation runner failed."
        )

        print(
            f"\nEvaluation failed: {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )