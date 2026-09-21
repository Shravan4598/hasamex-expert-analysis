"""
Free-form Q&A UI for the Hasamex Expert Analysis application.

This module provides the Streamlit interface for asking natural-language
questions across the loaded expert transcript corpus.

The UI delegates all retrieval, reranking, evidence verification, and
LLM generation to TranscriptQA.

The UI itself never:
    - invents transcript content,
    - generates timestamps,
    - assigns expert attribution,
    - performs unsupported reasoning,
    - creates citations independently.

All answer evidence comes from the analysis layer.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from logger import logging

from src.analysis.qa import TranscriptQA
from src.models import Evidence, GroundedAnswer

logger = logging.getLogger(__name__)


def render_qa(
    qa_service: TranscriptQA,
    *,
    experts: list[str] | None = None,
    markets: list[str] | None = None,
) -> None:
    """
    Render the free-form transcript Q&A page.

    Args:
        qa_service:
            Initialized TranscriptQA service.

        experts:
            Optional list of expert names for filtering.

        markets:
            Optional list of markets for filtering.
    """
    st.title("💬 Ask Questions")

    st.caption(
        "Ask questions across the available expert transcripts "
        "using source-grounded retrieval."
    )

    _render_grounding_notice()

    st.markdown("### Ask the Transcript Corpus")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What factors do the experts identify as "
            "important for robotic surgery adoption?"
        ),
        height=110,
        key="qa_question",
    )

    filter_columns = st.columns(2)

    with filter_columns[0]:
        selected_expert = _render_expert_filter(
            experts
        )

    with filter_columns[1]:
        selected_market = _render_market_filter(
            markets
        )

    _render_example_questions()

    ask_clicked = st.button(
        "Ask Question",
        type="primary",
        use_container_width=True,
        key="qa_ask_button",
    )

    if ask_clicked:
        _run_question(
            qa_service=qa_service,
            question=question,
            expert_name=selected_expert,
            market=selected_market,
        )

    answer = st.session_state.get(
        "hasamex_qa_answer"
    )

    if isinstance(
        answer,
        GroundedAnswer,
    ):
        st.divider()

        render_qa_answer(
            answer
        )


def render_qa_answer(
    answer: GroundedAnswer,
) -> None:
    """
    Render a complete grounded Q&A result.
    """
    st.markdown("### Answer")

    if not answer.evidence_sufficient:
        _render_insufficient_answer(
            answer
        )
        return

    st.markdown(
        answer.answer
    )

    _render_answer_metrics(
        answer
    )

    if answer.citations:
        _render_citations(
            answer
        )

    if answer.evidence:
        st.markdown(
            "### Supporting Evidence"
        )

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


def render_qa_history() -> None:
    """
    Render previously asked questions stored in the current session.

    The history is session-local and is not persisted as application
    data.
    """
    history = st.session_state.get(
        "hasamex_qa_history",
        [],
    )

    if not history:
        st.info(
            "No questions have been asked in this session."
        )
        return

    st.subheader(
        "Question History"
    )

    for index, item in enumerate(
        reversed(history),
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        question = item.get(
            "question",
            "Unknown question",
        )

        answer = item.get(
            "answer",
        )

        with st.expander(
            f"{index}. {question}",
            expanded=False,
        ):
            if isinstance(
                answer,
                GroundedAnswer,
            ):
                render_qa_answer(
                    answer
                )
            else:
                st.info(
                    "The stored answer is no longer available."
                )


def clear_qa_history() -> None:
    """
    Clear session-local Q&A history.
    """
    st.session_state[
        "hasamex_qa_history"
    ] = []

    st.session_state[
        "hasamex_qa_answer"
    ] = None


def _run_question(
    qa_service: TranscriptQA,
    question: str,
    expert_name: str | None,
    market: str | None,
) -> None:
    """
    Execute the Q&A service and store the result.
    """
    cleaned_question = _clean_question(
        question
    )

    if not cleaned_question:
        st.warning(
            "Please enter a question before asking."
        )
        return

    try:
        with st.spinner(
            "Searching transcript evidence and generating a grounded answer..."
        ):
            answer = qa_service.ask(
                question=cleaned_question,
                expert_name=expert_name,
                market=market,
            )

        st.session_state[
            "hasamex_qa_answer"
        ] = answer

        _append_to_history(
            question=cleaned_question,
            answer=answer,
        )

        if answer.evidence_sufficient:
            st.success(
                "Grounded answer generated."
            )
        else:
            st.warning(
                "The available evidence was insufficient for "
                "a reliable answer."
            )

    except Exception as error:
        logger.exception(
            "Free-form Q&A failed."
        )

        st.session_state[
            "hasamex_qa_answer"
        ] = None

        st.error(
            _safe_error_message(error)
        )


def _append_to_history(
    question: str,
    answer: GroundedAnswer,
) -> None:
    """
    Store the current question and answer in session state.
    """
    history = st.session_state.setdefault(
        "hasamex_qa_history",
        [],
    )

    history.append(
        {
            "question": question,
            "answer": answer,
        }
    )

    # Keep session state bounded.
    max_history = 20

    if len(history) > max_history:
        del history[:-max_history]


def _render_expert_filter(
    experts: list[str] | None,
) -> str | None:
    """
    Render the optional expert filter.
    """
    options = ["All experts"]

    if experts:
        options.extend(
            sorted(
                {
                    expert.strip()
                    for expert in experts
                    if isinstance(
                        expert,
                        str,
                    )
                    and expert.strip()
                },
                key=str.casefold,
            )
        )

    selected = st.selectbox(
        "Expert filter",
        options=options,
        key="qa_expert_filter",
    )

    if selected == "All experts":
        return None

    return selected


def _render_market_filter(
    markets: list[str] | None,
) -> str | None:
    """
    Render the optional market filter.
    """
    options = ["All markets"]

    if markets:
        options.extend(
            sorted(
                {
                    market.strip()
                    for market in markets
                    if isinstance(
                        market,
                        str,
                    )
                    and market.strip()
                },
                key=str.casefold,
            )
        )

    selected = st.selectbox(
        "Market filter",
        options=options,
        key="qa_market_filter",
    )

    if selected == "All markets":
        return None

    return selected


def _render_example_questions() -> None:
    """
    Display example questions that help users understand the Q&A
    workflow without forcing a particular question.
    """
    with st.expander(
        "Example questions",
        expanded=False,
    ):
        examples = [
            (
                "Adoption",
                "How would the experts describe current adoption "
                "of robotic surgery?"
            ),
            (
                "Barriers",
                "What are the main barriers to adoption?"
            ),
            (
                "Economics",
                "How important are hospital budgets and ROI "
                "in purchasing decisions?"
            ),
            (
                "Training",
                "What do the experts say about surgeon training "
                "and clinical outcomes?"
            ),
            (
                "Outlook",
                "What adoption trend do the experts expect "
                "over the next 3–5 years?"
            ),
            (
                "Timeline",
                "What purchasing timeline do the experts describe "
                "for a new robotic system?"
            ),
        ]

        for label, question in examples:
            st.markdown(
                f"**{label}:** {question}"
            )


def _render_answer_metrics(
    answer: GroundedAnswer,
) -> None:
    """
    Render descriptive answer-quality metadata.
    """
    columns = st.columns(3)

    verified_count = sum(
        1
        for evidence in answer.evidence
        if evidence.status.value == "verified"
    )

    with columns[0]:
        st.metric(
            "Confidence",
            f"{answer.confidence:.0%}",
            help=(
                "Confidence returned by the grounded analysis "
                "pipeline."
            ),
        )

    with columns[1]:
        st.metric(
            "Evidence Coverage",
            f"{answer.evidence_coverage:.0%}",
            help=(
                "Share of returned evidence items that passed "
                "the verification step."
            ),
        )

    with columns[2]:
        st.metric(
            "Verified Evidence",
            verified_count,
            help=(
                "Number of evidence items marked verified "
                "by the evidence pipeline."
            ),
        )


def _render_insufficient_answer(
    answer: GroundedAnswer,
) -> None:
    """
    Render a fail-closed response.
    """
    st.warning(
        "Insufficient transcript evidence"
    )

    st.markdown(
        answer.answer
    )

    if answer.refusal_reason:
        st.markdown(
            f"**Reason:** {answer.refusal_reason}"
        )

    st.caption(
        "The system intentionally does not fill evidence gaps "
        "with unsupported information."
    )


def _render_citations(
    answer: GroundedAnswer,
) -> None:
    """
    Render source citations associated with the answer.
    """
    st.markdown(
        "### Sources"
    )

    for citation in answer.citations:
        label = getattr(
            citation,
            "label",
            None,
        )

        source_file = getattr(
            citation,
            "source_file",
            None,
        )

        start_timestamp = getattr(
            citation,
            "start_timestamp",
            None,
        )

        end_timestamp = getattr(
            citation,
            "end_timestamp",
            None,
        )

        source_label = (
            label
            or source_file
            or "Unknown source"
        )

        timestamp = _format_timestamp_range(
            start_timestamp,
            end_timestamp,
        )

        st.markdown(
            f"- **{source_label}** — `{timestamp}`"
        )


def _render_evidence(
    evidence: Evidence,
    *,
    index: int,
) -> None:
    """
    Render one supporting evidence item.
    """
    expert = (
        evidence.expert_name
        or "Unknown expert"
    )

    source = (
        evidence.source_file
        or "Unknown source"
    )

    timestamp = _format_timestamp_range(
        evidence.start_timestamp,
        evidence.end_timestamp,
    )

    with st.expander(
        f"Evidence {index} · {expert} · {timestamp}",
        expanded=False,
    ):
        st.markdown(
            f"**Expert:** {expert}"
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
            f"**Source:** `{source}`"
        )

        st.markdown(
            f"**Timestamp:** `{timestamp}`"
        )

        if evidence.status.value == "verified":
            st.success(
                "Verified source evidence"
            )

            if evidence.quote:
                st.markdown(
                    f'> "{evidence.quote}"'
                )
        else:
            st.warning(
                "This evidence is not marked as an exact verified quote."
            )

            if evidence.verification_message:
                st.caption(
                    evidence.verification_message
                )

        st.caption(
            f"Evidence ID: {evidence.evidence_id}"
        )

        if evidence.chunk_id:
            st.caption(
                f"Chunk ID: {evidence.chunk_id}"
            )


def _render_grounding_notice() -> None:
    """
    Explain the grounding guarantees of the Q&A workflow.
    """
    st.info(
        """
        **Source-grounded Q&A**

        Your question is used to retrieve relevant transcript
        evidence before an answer is generated. Expert attribution,
        timestamps, and source information come from the transcript
        metadata.

        If the retrieved evidence is not sufficient, the system
        returns an evidence-limited response instead of guessing.
        """
    )


def _clean_question(
    question: str,
) -> str:
    """
    Normalize user-entered question text.
    """
    if not isinstance(
        question,
        str,
    ):
        return ""

    return " ".join(
        question.split()
    ).strip()


def _format_timestamp_range(
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> str:
    """
    Format an optional source timestamp range.
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


def _safe_error_message(
    error: Exception,
) -> str:
    """
    Return a UI-safe exception message.

    Detailed tracebacks are retained in the application logs.
    """
    message = str(error).strip()

    if message:
        return message

    return (
        "The question could not be answered. "
        "Check the application logs for details."
    )


__all__ = [
    "clear_qa_history",
    "render_qa",
    "render_qa_answer",
    "render_qa_history",
]