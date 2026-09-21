"""
Evaluation runner for the Hasamex Expert Analysis application.

Usage:

    python scripts/evaluate.py

Optional:

    python scripts/evaluate.py --cases evaluation/questions.json
    python scripts/evaluate.py --output evaluation/results.json
    python scripts/evaluate.py --verbose

Evaluation flow:

    Evaluation cases
          ↓
    TranscriptQA / analysis pipeline
          ↓
    Grounded answer
          ↓
    Evidence + citation checks
          ↓
    EvaluationResult
          ↓
    Console report / JSON report

Important:
    This evaluator does not invent reference answers or expected scores.
    Evaluation cases must explicitly define their expected evidence,
    source, expert, timestamp, or answer criteria.

The evaluator is intentionally conservative. A response without
traceable evidence is treated as a failed grounded-answer result rather
than being awarded credit based on semantic similarity alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger import logging
from src.analysis.disagreements import DisagreementAnalyzer
from src.analysis.interview_guide import InterviewGuideAnalyzer
from src.analysis.llm import GeminiLLM
from src.analysis.qa import TranscriptQA
from src.analysis.themes import ThemeAnalyzer
from src.config import Settings, get_settings
from src.models import (
    EvaluationCase,
    GroundedAnswer,
)
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when the evaluation pipeline cannot be executed."""


@dataclass(frozen=True)
class EvaluationServices:
    """
    Shared services used by the evaluation runner.
    """

    qa: TranscriptQA
    interview_guide: InterviewGuideAnalyzer
    themes: ThemeAnalyzer
    disagreements: DisagreementAnalyzer


@dataclass(frozen=True)
class CaseEvaluation:
    """
    Internal evaluation record for one test case.
    """

    case_id: str
    question: str
    passed: bool
    grounded: bool
    evidence_count: int
    citation_count: int
    checks: dict[str, bool]
    details: list[str]
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to JSON-serializable data.
        """
        return asdict(self)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the evaluation suite.

    Args:
        argv:
            Optional command-line arguments.

    Returns:
        Process exit code.
    """
    args = _parse_arguments(argv)

    try:
        settings = get_settings()
        settings.ensure_directories()

        cases_path = _resolve_cases_path(
            settings=settings,
            explicit_path=args.cases,
        )

        cases = _load_evaluation_cases(
            cases_path
        )

        if not cases:
            raise EvaluationError(
                f"No evaluation cases found in '{cases_path}'."
            )

        services = _build_services(
            settings
        )

        print(
            "=" * 76
        )
        print(
            "Hasamex Expert Analysis - Evaluation"
        )
        print(
            "=" * 76
        )

        print(
            f"\nEvaluation cases : {len(cases)}"
        )
        print(
            f"Cases file       : {cases_path}"
        )

        results = _evaluate_cases(
            cases=cases,
            services=services,
            verbose=args.verbose,
        )

        summary = _build_summary(
            results
        )

        _print_report(
            results=results,
            summary=summary,
        )

        if args.output:
            output_path = _resolve_output_path(
                settings=settings,
                output_path=args.output,
            )

            _write_results(
                output_path=output_path,
                results=results,
                summary=summary,
            )

            print(
                f"\nDetailed results saved to: {output_path}"
            )

        return 0 if summary["failed"] == 0 else 1

    except KeyboardInterrupt:
        logger.warning(
            "Evaluation interrupted by user."
        )

        print(
            "\nEvaluation interrupted."
        )

        return 130

    except Exception as error:
        logger.exception(
            "Evaluation failed."
        )

        print(
            "\nERROR: Evaluation failed."
        )
        print(
            f"Details: {error}"
        )

        return 1


def _build_services(
    settings: Settings,
) -> EvaluationServices:
    """
    Initialize the same core analysis services used by the application.
    """
    logger.info(
        "Initializing evaluation services."
    )

    vector_store = FAISSVectorStore(
        storage_dir=settings.vector_store_path,
        embedding_model=settings.embedding_model,
    )

    if not vector_store.exists():
        raise EvaluationError(
            "The vector store does not exist. "
            "Run 'python scripts/ingest.py' before evaluation."
        )

    vector_store.load()

    embeddings = EmbeddingService(
        model_name=settings.embedding_model,
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_service=embeddings,
        top_k=settings.retrieval_top_k,
        min_score=settings.min_retrieval_score,
    )

    reranker = Reranker(
        top_k=settings.rerank_top_k,
    )

    llm = GeminiLLM(
        api_key=settings.google_api_key,
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )

    interview_guide = InterviewGuideAnalyzer(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        min_evidence_coverage=(
            settings.min_evidence_coverage
        ),
    )

    themes = ThemeAnalyzer(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        min_evidence_coverage=(
            settings.min_evidence_coverage
        ),
    )

    disagreements = DisagreementAnalyzer(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        min_evidence_coverage=(
            settings.min_evidence_coverage
        ),
    )

    qa = TranscriptQA(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        min_evidence_coverage=(
            settings.min_evidence_coverage
        ),
    )

    return EvaluationServices(
        qa=qa,
        interview_guide=interview_guide,
        themes=themes,
        disagreements=disagreements,
    )


def _evaluate_cases(
    *,
    cases: list[EvaluationCase],
    services: EvaluationServices,
    verbose: bool,
) -> list[CaseEvaluation]:
    """
    Evaluate all cases independently.
    """
    results: list[CaseEvaluation] = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"\n[{index}/{len(cases)}] "
            f"{case.case_id}: {case.question}"
        )

        result = _evaluate_case(
            case=case,
            services=services,
            verbose=verbose,
        )

        results.append(
            result
        )

        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print(
            f"  {status} | "
            f"evidence={result.evidence_count} | "
            f"citations={result.citation_count}"
        )

    return results


def _evaluate_case(
    *,
    case: EvaluationCase,
    services: EvaluationServices,
    verbose: bool,
) -> CaseEvaluation:
    """
    Evaluate one case using grounded-answer checks.
    """
    checks: dict[str, bool] = {}
    details: list[str] = []

    try:
        answer = _run_case(
            case=case,
            services=services,
        )

    except Exception as error:
        logger.exception(
            "Evaluation case failed: %s",
            case.case_id,
        )

        return CaseEvaluation(
            case_id=case.case_id,
            question=case.question,
            passed=False,
            grounded=False,
            evidence_count=0,
            citation_count=0,
            checks={
                "execution": False,
            },
            details=[
                f"Case execution failed: {error}"
            ],
            answer=None,
        )

    evidence_count = len(
        answer.evidence
    )

    citation_count = len(
        answer.citations
    )

    checks["execution"] = True

    checks["evidence_present"] = (
        evidence_count > 0
    )

    checks["citations_present"] = (
        citation_count > 0
    )

    checks["evidence_sufficient"] = (
        answer.evidence_sufficient
    )

    checks["confidence_present"] = (
        answer.confidence is not None
    )

    checks["coverage_present"] = (
        answer.evidence_coverage is not None
    )

    checks["quotes_traceable"] = (
        _all_evidence_traceable(
            answer
        )
    )

    checks["timestamps_present"] = (
        _all_evidence_timestamps_present(
            answer
        )
    )

    checks["sources_present"] = (
        _all_evidence_sources_present(
            answer
        )
    )

    _append_failed_check_details(
        checks=checks,
        details=details,
    )

    expected_checks = _evaluate_expected_criteria(
        case=case,
        answer=answer,
    )

    checks.update(
        expected_checks
    )

    _append_failed_check_details(
        checks=expected_checks,
        details=details,
    )

    passed = all(
        checks.values()
    )

    if verbose:
        print(
            f"  Answer: "
            f"{_shorten(answer.answer, 220)}"
        )

        print(
            f"  Evidence sufficient: "
            f"{answer.evidence_sufficient}"
        )

        print(
            f"  Evidence coverage: "
            f"{answer.evidence_coverage}"
        )

        print(
            f"  Confidence: "
            f"{answer.confidence}"
        )

        if answer.refusal_reason:
            print(
                f"  Refusal reason: "
                f"{answer.refusal_reason}"
            )

    return CaseEvaluation(
        case_id=case.case_id,
        question=case.question,
        passed=passed,
        grounded=(
            checks["evidence_present"]
            and checks["evidence_sufficient"]
        ),
        evidence_count=evidence_count,
        citation_count=citation_count,
        checks=checks,
        details=details,
        answer=answer.answer,
    )


def _run_case(
    *,
    case: EvaluationCase,
    services: EvaluationServices,
) -> GroundedAnswer:
    """
    Route an evaluation case to the appropriate analysis workflow.
    """
    case_type = (
        getattr(
            case,
            "case_type",
            None,
        )
        or getattr(
            case,
            "type",
            None,
        )
        or "qa"
    )

    normalized_type = (
        str(case_type)
        .strip()
        .casefold()
    )

    if normalized_type in {
        "qa",
        "question",
        "free_form",
        "free-form",
    }:
        return services.qa.ask_query(
            question=case.question,
            expert_name=_get_optional_field(
                case,
                "expert_name",
            ),
            market=_get_optional_field(
                case,
                "market",
            ),
            source_file=_get_optional_field(
                case,
                "source_file",
            ),
        )

    if normalized_type in {
        "interview",
        "interview_guide",
        "interview-guide",
    }:
        return services.interview_guide.answer_question(
            question=case.question,
            expert_name=_get_optional_field(
                case,
                "expert_name",
            ),
        )

    raise EvaluationError(
        f"Unsupported evaluation case type "
        f"'{case_type}' for case '{case.case_id}'."
    )


def _evaluate_expected_criteria(
    *,
    case: EvaluationCase,
    answer: GroundedAnswer,
) -> dict[str, bool]:
    """
    Evaluate explicitly supplied expectations.

    Supported expectation fields are intentionally conservative:

        expected_expert
        expected_market
        expected_source
        expected_timestamp
        expected_keywords
        expected_quote_fragments

    Missing expectations are not treated as failures.
    """
    checks: dict[str, bool] = {}

    expected_expert = _get_optional_field(
        case,
        "expected_expert",
    )

    if expected_expert:
        checks["expected_expert"] = _contains_evidence_value(
            answer,
            expected_expert,
            attribute="expert_name",
        )

    expected_market = _get_optional_field(
        case,
        "expected_market",
    )

    if expected_market:
        checks["expected_market"] = _contains_evidence_value(
            answer,
            expected_market,
            attribute="market",
        )

    expected_source = _get_optional_field(
        case,
        "expected_source",
    )

    if expected_source:
        checks["expected_source"] = _contains_evidence_value(
            answer,
            expected_source,
            attribute="source_file",
        )

    expected_timestamp = _get_optional_field(
        case,
        "expected_timestamp",
    )

    if expected_timestamp:
        checks["expected_timestamp"] = (
            _contains_timestamp(
                answer,
                expected_timestamp,
            )
        )

    expected_keywords = _get_string_list(
        case,
        "expected_keywords",
    )

    if expected_keywords:
        answer_text = (
            answer.answer or ""
        ).casefold()

        checks["expected_keywords"] = all(
            keyword.casefold()
            in answer_text
            for keyword in expected_keywords
        )

    expected_fragments = _get_string_list(
        case,
        "expected_quote_fragments",
    )

    if expected_fragments:
        evidence_quotes = [
            evidence.quote.casefold()
            for evidence in answer.evidence
            if evidence.quote
        ]

        checks["expected_quote_fragments"] = all(
            any(
                fragment.casefold()
                in quote
                for quote in evidence_quotes
            )
            for fragment in expected_fragments
        )

    return checks


def _all_evidence_traceable(
    answer: GroundedAnswer,
) -> bool:
    """
    Verify that every displayed evidence item has source content.
    """
    if not answer.evidence:
        return False

    for evidence in answer.evidence:
        if not evidence.quote:
            return False

        if not evidence.source_text:
            return False

        if not evidence.source_file:
            return False

    return True


def _all_evidence_timestamps_present(
    answer: GroundedAnswer,
) -> bool:
    """
    Verify that evidence contains usable timestamp information.
    """
    if not answer.evidence:
        return False

    return all(
        evidence.start_timestamp
        or evidence.end_timestamp
        for evidence in answer.evidence
    )


def _all_evidence_sources_present(
    answer: GroundedAnswer,
) -> bool:
    """
    Verify that each evidence item has source metadata.
    """
    if not answer.evidence:
        return False

    return all(
        evidence.source_file
        and evidence.expert_name
        for evidence in answer.evidence
    )


def _contains_evidence_value(
    answer: GroundedAnswer,
    expected: str,
    *,
    attribute: str,
) -> bool:
    """
    Check whether an expected metadata value appears in evidence.
    """
    expected_normalized = expected.strip().casefold()

    if not expected_normalized:
        return True

    for evidence in answer.evidence:
        value = getattr(
            evidence,
            attribute,
            None,
        )

        if (
            isinstance(value, str)
            and expected_normalized
            in value.casefold()
        ):
            return True

    return False


def _contains_timestamp(
    answer: GroundedAnswer,
    expected_timestamp: str,
) -> bool:
    """
    Check whether the expected timestamp appears in evidence.
    """
    expected = (
        expected_timestamp
        .strip()
        .casefold()
    )

    if not expected:
        return True

    for evidence in answer.evidence:
        timestamps = (
            evidence.start_timestamp,
            evidence.end_timestamp,
        )

        if any(
            isinstance(timestamp, str)
            and expected in timestamp.casefold()
            for timestamp in timestamps
        ):
            return True

    return False


def _append_failed_check_details(
    *,
    checks: dict[str, bool],
    details: list[str],
) -> None:
    """
    Add human-readable explanations for failed checks.
    """
    for check_name, passed in checks.items():
        if not passed:
            details.append(
                f"Failed check: {check_name}"
            )


def _build_summary(
    results: list[CaseEvaluation],
) -> dict[str, Any]:
    """
    Build aggregate evaluation statistics.

    These are descriptive measurements of the executed test suite.
    They are not model benchmarks and must not be interpreted as
    general accuracy claims beyond the supplied evaluation cases.
    """
    total = len(results)

    passed = sum(
        result.passed
        for result in results
    )

    failed = total - passed

    grounded = sum(
        result.grounded
        for result in results
    )

    evidence_cases = sum(
        result.evidence_count > 0
        for result in results
    )

    citation_cases = sum(
        result.citation_count > 0
        for result in results
    )

    pass_rate = (
        passed / total
        if total
        else 0.0
    )

    grounding_rate = (
        grounded / total
        if total
        else 0.0
    )

    evidence_rate = (
        evidence_cases / total
        if total
        else 0.0
    )

    citation_rate = (
        citation_cases / total
        if total
        else 0.0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "grounded_cases": grounded,
        "grounding_rate": grounding_rate,
        "cases_with_evidence": evidence_cases,
        "evidence_rate": evidence_rate,
        "cases_with_citations": citation_cases,
        "citation_rate": citation_rate,
    }


def _print_report(
    *,
    results: list[CaseEvaluation],
    summary: dict[str, Any],
) -> None:
    """
    Print the evaluation report.
    """
    print(
        "\n" + "=" * 76
    )
    print(
        "EVALUATION SUMMARY"
    )
    print(
        "=" * 76
    )

    print(
        f"Total cases          : {summary['total']}"
    )

    print(
        f"Passed               : {summary['passed']}"
    )

    print(
        f"Failed               : {summary['failed']}"
    )

    print(
        f"Pass rate            : "
        f"{summary['pass_rate']:.2%}"
    )

    print(
        f"Grounded cases       : "
        f"{summary['grounded_cases']}"
    )

    print(
        f"Grounding rate       : "
        f"{summary['grounding_rate']:.2%}"
    )

    print(
        f"Cases with evidence  : "
        f"{summary['cases_with_evidence']}"
    )

    print(
        f"Evidence rate        : "
        f"{summary['evidence_rate']:.2%}"
    )

    print(
        f"Cases with citations  : "
        f"{summary['cases_with_citations']}"
    )

    print(
        f"Citation rate        : "
        f"{summary['citation_rate']:.2%}"
    )

    print(
        "\nCase details:"
    )

    for result in results:
        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print(
            f"\n[{status}] {result.case_id}"
        )

        print(
            f"  Question: {result.question}"
        )

        print(
            f"  Evidence: {result.evidence_count}"
        )

        print(
            f"  Citations: {result.citation_count}"
        )

        if result.details:
            for detail in result.details:
                print(
                    f"  - {detail}"
                )

    print(
        "\n" + "=" * 76
    )

    print(
        "Evaluation completed."
    )

    print(
        "=" * 76
    )


def _load_evaluation_cases(
    path: Path,
) -> list[EvaluationCase]:
    """
    Load evaluation cases from JSON.
    """
    if not path.exists():
        raise EvaluationError(
            f"Evaluation case file does not exist: {path}"
        )

    if not path.is_file():
        raise EvaluationError(
            f"Evaluation case path is not a file: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

    except json.JSONDecodeError as error:
        raise EvaluationError(
            f"Invalid JSON in evaluation file '{path}': "
            f"{error}"
        ) from error

    except OSError as error:
        raise EvaluationError(
            f"Could not read evaluation file '{path}'."
        ) from error

    if isinstance(
        payload,
        dict,
    ):
        raw_cases = payload.get(
            "cases",
            [],
        )

    elif isinstance(
        payload,
        list,
    ):
        raw_cases = payload

    else:
        raise EvaluationError(
            "Evaluation JSON must contain either a list of cases "
            "or an object with a 'cases' list."
        )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise EvaluationError(
            "The 'cases' field must be a list."
        )

    cases: list[EvaluationCase] = []

    for index, item in enumerate(
        raw_cases,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise EvaluationError(
                f"Evaluation case #{index} must be a JSON object."
            )

        try:
            cases.append(
                EvaluationCase.model_validate(
                    item
                )
            )

        except Exception as error:
            raise EvaluationError(
                f"Invalid evaluation case #{index}: {error}"
            ) from error

    return cases


def _resolve_cases_path(
    *,
    settings: Settings,
    explicit_path: str | None,
) -> Path:
    """
    Resolve the evaluation-case file.
    """
    if explicit_path:
        path = Path(
            explicit_path
        ).expanduser().resolve()

    else:
        path = (
            Path.cwd()
            / "evaluation"
            / "questions.json"
        )

    return path


def _resolve_output_path(
    *,
    settings: Settings,
    output_path: str,
) -> Path:
    """
    Resolve and prepare the output path.
    """
    path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _write_results(
    *,
    output_path: Path,
    results: list[CaseEvaluation],
    summary: dict[str, Any],
) -> None:
    """
    Persist detailed evaluation results as JSON.
    """
    payload = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": summary,
        "cases": [
            result.to_dict()
            for result in results
        ],
    }

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as error:
        raise EvaluationError(
            f"Could not write evaluation results "
            f"to '{output_path}'."
        ) from error


def _get_optional_field(
    model: Any,
    field_name: str,
) -> str | None:
    """
    Safely read an optional string field from a Pydantic model.
    """
    value = getattr(
        model,
        field_name,
        None,
    )

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        return str(value)

    value = value.strip()

    return value or None


def _get_string_list(
    model: Any,
    field_name: str,
) -> list[str]:
    """
    Safely read a list of strings from an evaluation model.
    """
    value = getattr(
        model,
        field_name,
        None,
    )

    if not value:
        return []

    if isinstance(
        value,
        str,
    ):
        return [
            value.strip()
        ] if value.strip() else []

    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        item.strip()
        for item in value
        if isinstance(
            item,
            str,
        )
        and item.strip()
    ]


def _shorten(
    text: str | None,
    maximum_length: int,
) -> str:
    """
    Shorten console output without modifying the underlying answer.
    """
    if not text:
        return ""

    normalized = " ".join(
        text.split()
    )

    if len(normalized) <= maximum_length:
        return normalized

    return (
        normalized[: maximum_length - 3]
        + "..."
    )


def _parse_arguments(
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grounded transcript analysis "
            "against explicit evaluation cases."
        )
    )

    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help=(
            "Path to evaluation/questions.json. "
            "Defaults to evaluation/questions.json."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path for detailed JSON evaluation results."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print generated answers and additional evaluation details."
        ),
    )

    return parser.parse_args(
        argv
    )


if __name__ == "__main__":
    sys.exit(
        main()
    )