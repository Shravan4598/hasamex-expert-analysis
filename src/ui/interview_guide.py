"""
Interview-guide UI for the Hasamex Expert Analysis application.

This module renders the six predefined interview-guide questions and
their source-grounded answers for the available experts.

The UI layer does not perform retrieval or LLM reasoning itself.
It delegates analysis to InterviewGuideAnalyzer and only presents
the returned structured results.

Key principles:

    - No hard-coded expert answers.
    - No generated timestamps.
    - Evidence metadata comes from the source pipeline.
    - Exact quotes are shown only when marked as verified.
    - Insufficient evidence is presented explicitly.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from logger import logging
from src.analysis.interview_guide import (
    INTERVIEW_QUESTIONS,
    InterviewGuideAnalyzer,
)
from src.models import (
    Evidence,
    GroundedAnswer,
    InterviewQuestion,
)

logger = logging.getLogger(__name__)


def render_interview_guide(
    analyzer: InterviewGuideAnalyzer,
    *,
    experts: list[str] | None = None,
) -> None:
    """
    Render the interview-guide analysis page.

    Args:
        analyzer:
            Initialized InterviewGuideAnalyzer instance.

        experts:
            Optional list of expert names. If omitted, experts are
            discovered from the analyzer's available corpus.
    """
    st.title("📋 Interview Guide Analysis")

    st.caption(
        "Source-grounded answers to the predefined interview questions."
    )

    _render_grounding_notice()

    questions = analyzer.get_questions()

    if not questions:
        st.warning(
            "No interview-guide questions are configured."
        )
        return

    available_experts = (
        experts
        if experts is not None
        else _discover_experts(analyzer)
    )

    if not available_experts:
        st.info(
            "No experts are currently available in the loaded corpus."
        )
        return

    selected_expert = st.selectbox(
        "Select expert",
        options=available_experts,
        key="interview_selected_expert",
    )

    selected_question_id = _render_question_selector(
        questions
    )

    selected_question = _find_question(
        questions,
        selected_question_id,
    )

    if selected_question is None:
        st.error(
            "The selected interview question could not be found."
        )
        return

    _render_question_header(
        selected_question
    )

    if st.button(
        "Generate Grounded Answer",
        type="primary",
        use_container_width=True,
        key="generate_interview_answer",
    ):
        _run_single_analysis(
            analyzer=analyzer,
            question=selected_question,
            expert_name=selected_expert,
        )

    cached_answer = _get_cached_answer(
        expert_name=selected_expert,
        question_id=selected_question.question_id,
    )

    if cached_answer is not None:
        st.divider()
        render_grounded_answer(
            cached_answer
        )


def render_all_experts_question(
    analyzer: InterviewGuideAnalyzer,
    question: InterviewQuestion,
    experts: list[str],
) -> None:
    """
    Render one interview question across multiple experts.

    This is useful for side-by-side comparison in the UI.
    """
    st.subheader(
        question.question
    )

    if not experts:
        st.info(
            "No experts are available for comparison."
        )
        return

    for expert_name in experts:
        with st.expander(
            expert_name,
            expanded=False,
        ):
            cache_key = _answer_cache_key(
                expert_name,
                question.question_id,
            )

            answer = st.session_state.get(
                cache_key
            )

            if answer is None:
                if st.button(
                    f"Analyze {expert_name}",
                    key=f"analyze_{cache_key}",
                ):
                    try:
                        with st.spinner(
                            "Retrieving transcript evidence..."
                        ):
                            answer = analyzer.analyze_question(
                                question=question.question_id,
                                expert_name=expert_name,
                            )

                        st.session_state[
                            cache_key
                        ] = answer

                        st.rerun()

                    except Exception as error:
                        logger.exception(
                            "Interview-guide analysis failed "
                            "for expert '%s'.",
                            expert_name,
                        )
                        st.error(
                            _safe_error_message(error)
                        )
            else:
                render_grounded_answer(
                    answer
                )


def render_grounded_answer(
    answer: GroundedAnswer,
) -> None:
    """
    Render a grounded answer with its evidence and citations.
    """
    if not answer.evidence_sufficient:
        _render_insufficient_evidence(
            answer
        )
        return

    _render_answer_summary(
        answer
    )

    if answer.citations:
        _render_citations(
            answer
        )

    if answer.evidence:
        st.markdown("### Supporting Evidence")

        for index, evidence in enumerate(
            answer.evidence,
            start=1,
        ):
            _render_evidence(
                evidence,
                index=index,
            )

    if answer.refusal_reason:
        st.caption(
            f"Grounding note: {answer.refusal_reason}"
        )


def render_question_list(
    analyzer: InterviewGuideAnalyzer,
) -> None:
    """
    Render the complete interview-guide question list.
    """
    st.subheader(
        "Interview Questions"
    )

    questions = analyzer.get_questions()

    if not questions:
        st.info(
            "No questions are configured."
        )
        return

    for question in questions:
        with st.expander(
            f"{question.question_id}: {question.question}",
            expanded=False,
        ):
            st.caption(
                f"Topic: {question.topic}"
            )


def render_question_comparison(
    analyzer: InterviewGuideAnalyzer,
    experts: list[str],
) -> None:
    """
    Render a complete expert comparison workflow.

    Users select one question and can analyze it independently for
    each available expert.
    """
    st.title(
        "📊 Expert Question Comparison"
    )

    questions = analyzer.get_questions()

    if not questions:
        st.warning(
            "No interview-guide questions are configured."
        )
        return

    if not experts:
        st.info(
            "No experts are available."
        )
        return

    question_options = {
        (
            f"{question.question_id}: "
            f"{question.question}"
        ): question
        for question in questions
    }

    selected_label = st.selectbox(
        "Interview question",
        options=list(question_options.keys()),
        key="comparison_question",
    )

    selected_question = question_options[
        selected_label
    ]

    if st.button(
        "Analyze All Experts",
        type="primary",
        use_container_width=True,
        key="analyze_all_experts",
    ):
        for expert_name in experts:
            try:
                with st.spinner(
                    f"Analyzing {expert_name}..."
                ):
                    answer = analyzer.analyze_question(
                        question=selected_question.question_id,
                        expert_name=expert_name,
                    )

                st.session_state[
                    _answer_cache_key(
                        expert_name,
                        selected_question.question_id,
                    )
                ] = answer

            except Exception as error:
                logger.exception(
                    "Failed to analyze '%s' for '%s'.",
                    selected_question.question_id,
                    expert_name,
                )

                st.error(
                    f"Could not analyze {expert_name}: "
                    f"{_safe_error_message(error)}"
                )

    st.divider()

    render_all_experts_question(
        analyzer=analyzer,
        question=selected_question,
        experts=experts,
    )


def _render_question_selector(
    questions: list[InterviewQuestion],
) -> str:
    """
    Render the interview-question selector.
    """
    labels = [
        (
            f"{question.question_id}: "
            f"{question.question}"
        )
        for question in questions
    ]

    selected_label = st.selectbox(
        "Interview question",
        options=labels,
        key="interview_question_selector",
    )

    return selected_label.split(
        ":",
        maxsplit=1,
    )[0].strip()


def _find_question(
    questions: list[InterviewQuestion],
    question_id: str,
) -> InterviewQuestion | None:
    """
    Find an interview question by stable ID.
    """
    for question in questions:
        if question.question_id == question_id:
            return question

    return None


def _render_question_header(
    question: InterviewQuestion,
) -> None:
    """
    Render the selected interview question.
    """
    st.markdown(
        f"### {question.question_id}"
    )

    st.markdown(
        f"**{question.question}**"
    )

    if question.topic:
        st.caption(
            f"Topic: {question.topic}"
        )


def _run_single_analysis(
    analyzer: InterviewGuideAnalyzer,
    question: InterviewQuestion,
    expert_name: str,
) -> None:
    """
    Execute one interview-guide analysis and cache the result.
    """
    try:
        with st.spinner(
            "Retrieving evidence and generating grounded answer..."
        ):
            answer = analyzer.analyze_question(
                question=question.question_id,
                expert_name=expert_name,
            )

        st.session_state[
            _answer_cache_key(
                expert_name,
                question.question_id,
            )
        ] = answer

        st.success(
            "Grounded analysis generated."
        )

    except Exception as error:
        logger.exception(
            "Interview-guide analysis failed for '%s' / '%s'.",
            expert_name,
            question.question_id,
        )

        st.error(
            _safe_error_message(error)
        )


def _get_cached_answer(
    expert_name: str,
    question_id: str,
) -> GroundedAnswer | None:
    """
    Retrieve a previously generated answer from Streamlit state.
    """
    value = st.session_state.get(
        _answer_cache_key(
            expert_name,
            question_id,
        )
    )

    if isinstance(
        value,
        GroundedAnswer,
    ):
        return value

    return None


def _answer_cache_key(
    expert_name: str,
    question_id: str,
) -> str:
    """
    Create a deterministic Streamlit session-state key.
    """
    safe_expert = (
        expert_name.strip()
        .lower()
        .replace(" ", "_")
    )

    safe_question = (
        question_id.strip()
        .lower()
    )

    return (
        f"interview_answer__"
        f"{safe_expert}__"
        f"{safe_question}"
    )


def _discover_experts(
    analyzer: InterviewGuideAnalyzer,
) -> list[str]:
    """
    Discover expert names from the analyzer's retrieval corpus.

    The retrieval layer owns the actual corpus metadata, so this
    function intentionally does not contain hard-coded expert names.
    """
    try:
        results = analyzer.retriever.retrieve(
            query="expert interview",
            top_k=50,
        )

        experts: set[str] = set()

        for result in results:
            expert_name = (
                result.chunk.expert_name
            )

            if expert_name:
                experts.add(
                    expert_name.strip()
                )

        return sorted(
            experts,
            key=str.casefold,
        )

    except Exception as error:
        logger.exception(
            "Failed to discover experts."
        )

        st.warning(
            "Expert discovery is currently unavailable."
        )

        return []


def _render_answer_summary(
    answer: GroundedAnswer,
) -> None:
    """
    Render the generated grounded answer and confidence metadata.
    """
    st.markdown("### Answer")

    st.markdown(
        answer.answer
    )

    columns = st.columns(3)

    with columns[0]:
        st.metric(
            "Confidence",
            f"{answer.confidence:.0%}",
        )

    with columns[1]:
        st.metric(
            "Evidence Coverage",
            f"{answer.evidence_coverage:.0%}",
        )

    with columns[2]:
        st.metric(
            "Verified Evidence",
            sum(
                1
                for evidence in answer.evidence
                if evidence.status.value == "verified"
            ),
        )


def _render_insufficient_evidence(
    answer: GroundedAnswer,
) -> None:
    """
    Render a fail-closed answer when evidence is insufficient.
    """
    st.warning(
        "Insufficient transcript evidence"
    )

    st.markdown(
        answer.answer
    )

    if answer.refusal_reason:
        st.caption(
            f"Reason: {answer.refusal_reason}"
        )

    st.caption(
        "The application intentionally avoids generating an "
        "unsupported answer."
    )


def _render_citations(
    answer: GroundedAnswer,
) -> None:
    """
    Render citation metadata associated with the answer.
    """
    st.markdown(
        "### Sources"
    )

    for citation in answer.citations:
        label = (
            citation.label
            if citation.label
            else citation.source_file
        )

        timestamp = _format_timestamp_range(
            citation.start_timestamp,
            citation.end_timestamp,
        )

        st.markdown(
            f"- **{label}** — {timestamp}"
        )


def _render_evidence(
    evidence: Evidence,
    *,
    index: int,
) -> None:
    """
    Render one evidence item.
    """
    timestamp = _format_timestamp_range(
        evidence.start_timestamp,
        evidence.end_timestamp,
    )

    expert = (
        evidence.expert_name
        or "Unknown speaker"
    )

    source = (
        evidence.source_file
        or "Unknown source"
    )

    status = evidence.status.value

    with st.expander(
        f"Evidence {index} · {expert} · {timestamp}",
        expanded=False,
    ):
        st.markdown(
            f"**Source:** `{source}`"
        )

        if evidence.market:
            st.markdown(
                f"**Market:** {evidence.market}"
            )

        if evidence.speaker:
            st.markdown(
                f"**Speaker:** {evidence.speaker}"
            )

        st.markdown(
            f"**Timestamp:** `{timestamp}`"
        )

        if status == "verified":
            st.success(
                "Quote verified against the available source text."
            )

            st.markdown(
                f'> "{evidence.quote}"'
            )
        else:
            st.warning(
                "This evidence was not verified as an exact quote "
                "and is therefore not displayed as a verified quote."
            )

            if evidence.verification_message:
                st.caption(
                    evidence.verification_message
                )

        if evidence.chunk_id:
            st.caption(
                f"Evidence ID: {evidence.evidence_id}  "
                f"• Chunk: {evidence.chunk_id}"
            )


def _format_timestamp_range(
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> str:
    """
    Format an evidence timestamp range safely.
    """
    if start_timestamp and end_timestamp:
        if start_timestamp == end_timestamp:
            return start_timestamp

        return (
            f"{start_timestamp} → {end_timestamp}"
        )

    if start_timestamp:
        return start_timestamp

    if end_timestamp:
        return end_timestamp

    return "Timestamp unavailable"


def _render_grounding_notice() -> None:
    """
    Display the interview-guide grounding policy.
    """
    st.info(
        """
        **Evidence-grounded analysis**

        Answers are generated from retrieved transcript evidence.
        Source files, expert attribution, and timestamps are taken
        from transcript metadata. If the evidence is insufficient,
        the system will say so instead of filling the gap with
        unsupported information.
        """
    )


def _safe_error_message(
    error: Exception,
) -> str:
    """
    Convert an exception into a UI-safe message.

    Detailed stack traces remain in the application logs rather than
    being exposed to the user.
    """
    message = str(error).strip()

    if not message:
        return (
            "The analysis could not be completed. "
            "Check the application logs for details."
        )

    return message


__all__ = [
    "render_all_experts_question",
    "render_grounded_answer",
    "render_interview_guide",
    "render_question_comparison",
    "render_question_list",
]