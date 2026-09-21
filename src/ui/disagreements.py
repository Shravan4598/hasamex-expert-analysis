"""
Differences and disagreement UI for the Hasamex Expert Analysis app.

This module renders cross-expert differences identified by the analysis
layer.

The analysis layer determines whether evidence represents:

    - Direct disagreement
    - Different emphasis
    - Different experience
    - Complementary views
    - Insufficient evidence

This UI layer does not make those judgments itself. It only presents
the structured results together with the underlying evidence and source
metadata.

Important principles:

    - No hard-coded conclusions.
    - No invented expert statements.
    - No invented timestamps.
    - Source metadata comes from the transcript pipeline.
    - Evidence is displayed with expert attribution.
    - Insufficient evidence is explicitly surfaced.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import streamlit as st

from logger import logging

from src.analysis.disagreements import (
    DisagreementAnalyzer,
)
from src.models import (
    DifferenceType,
    Disagreement,
    Evidence,
)

logger = logging.getLogger(__name__)


def render_disagreements(
    analyzer: DisagreementAnalyzer,
) -> None:
    """
    Render the main cross-expert differences page.

    Args:
        analyzer:
            Initialized DisagreementAnalyzer instance.
    """
    st.title("⚖️ Differences & Disagreements")

    st.caption(
        "Compare expert perspectives using source-grounded evidence."
    )

    _render_grounding_notice()

    st.markdown("### Analysis Scope")

    question = st.text_input(
        "Optional focus question",
        placeholder=(
            "Example: What differences exist in views about "
            "robotic surgery adoption?"
        ),
        help=(
            "Leave this empty to analyze broad cross-expert "
            "differences. Enter a question to focus the analysis."
        ),
        key="disagreement_focus_question",
    )

    if st.button(
        "Analyze Expert Differences",
        type="primary",
        use_container_width=True,
        key="analyze_disagreements",
    ):
        _run_disagreement_analysis(
            analyzer=analyzer,
            question=question.strip() or None,
        )

    disagreements = st.session_state.get(
        "hasamex_disagreements"
    )

    if disagreements is not None:
        st.divider()

        render_disagreement_summary(
            disagreements
        )

        render_disagreements_list(
            disagreements
        )


def render_disagreements_list(
    disagreements: list[Disagreement],
) -> None:
    """
    Render all identified cross-expert differences.
    """
    if not disagreements:
        st.info(
            "No sufficiently supported cross-expert differences "
            "were identified."
        )
        return

    for index, disagreement in enumerate(
        disagreements,
        start=1,
    ):
        _render_disagreement(
            disagreement,
            index=index,
        )


def render_disagreement(
    disagreement: Disagreement,
    *,
    index: int | None = None,
) -> None:
    """
    Public helper for rendering a single disagreement/difference.
    """
    _render_disagreement(
        disagreement,
        index=index,
    )


def render_disagreement_summary(
    disagreements: list[Disagreement],
) -> None:
    """
    Render aggregate counts by difference type.

    This is descriptive only; it does not rank or score the
    differences.
    """
    if not disagreements:
        return

    st.subheader(
        "Analysis Summary"
    )

    counts = Counter(
        _difference_type_value(
            disagreement
        )
        for disagreement in disagreements
    )

    total = len(disagreements)

    columns = st.columns(
        min(max(len(counts), 1), 5)
    )

    for column, (
        difference_type,
        count,
    ) in zip(
        columns,
        sorted(
            counts.items(),
            key=lambda item: item[0],
        ),
    ):
        with column:
            st.metric(
                _format_difference_type(
                    difference_type
                ),
                count,
                help=(
                    f"{count} of {total} identified "
                    "difference(s) belong to this category."
                ),
            )

    st.caption(
        "Counts describe the categories produced by the analysis "
        "pipeline; they are not quality scores or rankings."
    )


def render_difference_type_guide() -> None:
    """
    Explain the meaning of the difference categories.
    """
    with st.expander(
        "How are differences classified?",
        expanded=False,
    ):
        definitions = {
            "Direct disagreement": (
                "Experts express materially conflicting positions "
                "on the same issue."
            ),
            "Different emphasis": (
                "Experts discuss the same issue but emphasize "
                "different factors or priorities."
            ),
            "Different experience": (
                "The difference is associated with different "
                "markets, roles, institutions, or observed experiences."
            ),
            "Complementary": (
                "The views address different aspects that can "
                "reasonably coexist rather than conflict."
            ),
            "Insufficient evidence": (
                "The available transcript evidence does not support "
                "a reliable comparison."
            ),
        }

        for label, description in definitions.items():
            st.markdown(
                f"**{label}:** {description}"
            )


def _run_disagreement_analysis(
    analyzer: DisagreementAnalyzer,
    question: str | None,
) -> None:
    """
    Execute disagreement analysis and cache the results.
    """
    try:
        with st.spinner(
            "Retrieving cross-expert evidence and analyzing differences..."
        ):
            if question:
                disagreements = (
                    analyzer.analyze_for_question(
                        question
                    )
                )
            else:
                disagreements = analyzer.analyze()

        st.session_state[
            "hasamex_disagreements"
        ] = disagreements

        if disagreements:
            st.success(
                "Cross-expert difference analysis completed."
            )
        else:
            st.info(
                "The available evidence did not support any "
                "cross-expert differences."
            )

    except Exception as error:
        logger.exception(
            "Disagreement analysis failed."
        )

        st.session_state[
            "hasamex_disagreements"
        ] = None

        st.error(
            _safe_error_message(error)
        )


def _render_disagreement(
    disagreement: Disagreement,
    *,
    index: int | None,
) -> None:
    """
    Render one cross-expert difference.
    """
    difference_type = _difference_type_value(
        disagreement
    )

    title = _build_title(
        disagreement,
        index=index,
    )

    with st.container(border=True):
        st.markdown(
            f"### {title}"
        )

        _render_difference_badge(
            difference_type
        )

        summary = getattr(
            disagreement,
            "summary",
            None,
        )

        if summary:
            st.markdown(
                summary
            )

        experts = _extract_experts(
            disagreement
        )

        if experts:
            st.markdown(
                "**Experts represented:** "
                + " • ".join(experts)
            )

        evidence = _extract_evidence(
            disagreement
        )

        if evidence:
            st.markdown(
                "#### Supporting Evidence"
            )

            for evidence_index, item in enumerate(
                evidence,
                start=1,
            ):
                _render_evidence(
                    item,
                    index=evidence_index,
                )
        else:
            st.warning(
                "No supporting evidence is attached to this finding."
            )

        disagreement_id = getattr(
            disagreement,
            "disagreement_id",
            None,
        )

        if disagreement_id:
            st.caption(
                f"Analysis ID: {disagreement_id}"
            )


def _render_difference_badge(
    difference_type: str,
) -> None:
    """
    Render a readable category indicator.
    """
    label = _format_difference_type(
        difference_type
    )

    if difference_type == (
        DifferenceType.DIRECT_DISAGREEMENT.value
    ):
        st.error(
            f"Classification: {label}"
        )

    elif difference_type == (
        DifferenceType.INSUFFICIENT_EVIDENCE.value
    ):
        st.warning(
            f"Classification: {label}"
        )

    elif difference_type == (
        DifferenceType.COMPLEMENTARY.value
    ):
        st.info(
            f"Classification: {label}"
        )

    else:
        st.info(
            f"Classification: {label}"
        )


def _render_evidence(
    evidence: Evidence,
    *,
    index: int,
) -> None:
    """
    Render one evidence item supporting a difference.
    """
    expert = (
        evidence.expert_name
        or "Unknown expert"
    )

    timestamp = _format_timestamp_range(
        evidence.start_timestamp,
        evidence.end_timestamp,
    )

    source = (
        evidence.source_file
        or "Unknown source"
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

        status = evidence.status.value

        if status == "verified":
            st.success(
                "Evidence verified against the available source text."
            )

            if evidence.quote:
                st.markdown(
                    f'> "{evidence.quote}"'
                )
        else:
            st.warning(
                "This item is not marked as a verified exact quote."
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


def _extract_experts(
    disagreement: Disagreement,
) -> list[str]:
    """
    Extract unique expert names from a disagreement.
    """
    experts: set[str] = set()

    direct_experts = getattr(
        disagreement,
        "experts",
        None,
    )

    if direct_experts:
        for expert in direct_experts:
            if isinstance(
                expert,
                str,
            ):
                cleaned = expert.strip()

                if cleaned:
                    experts.add(
                        cleaned
                    )

    for evidence in _extract_evidence(
        disagreement
    ):
        expert_name = getattr(
            evidence,
            "expert_name",
            None,
        )

        if expert_name:
            experts.add(
                expert_name.strip()
            )

    return sorted(
        experts,
        key=str.casefold,
    )


def _extract_evidence(
    disagreement: Disagreement,
) -> list[Evidence]:
    """
    Safely retrieve disagreement evidence.
    """
    evidence = getattr(
        disagreement,
        "evidence",
        None,
    )

    if not evidence:
        return []

    return list(evidence)


def _difference_type_value(
    disagreement: Disagreement,
) -> str:
    """
    Return the underlying string value of the difference type.
    """
    difference_type = getattr(
        disagreement,
        "difference_type",
        None,
    )

    if isinstance(
        difference_type,
        DifferenceType,
    ):
        return difference_type.value

    if difference_type is None:
        return DifferenceType.INSUFFICIENT_EVIDENCE.value

    return str(
        difference_type
    ).strip().lower()


def _format_difference_type(
    difference_type: str,
) -> str:
    """
    Convert an enum value into a human-readable label.
    """
    labels = {
        DifferenceType.DIRECT_DISAGREEMENT.value: (
            "Direct disagreement"
        ),
        DifferenceType.DIFFERENT_EMPHASIS.value: (
            "Different emphasis"
        ),
        DifferenceType.DIFFERENT_EXPERIENCE.value: (
            "Different experience"
        ),
        DifferenceType.COMPLEMENTARY.value: (
            "Complementary views"
        ),
        DifferenceType.INSUFFICIENT_EVIDENCE.value: (
            "Insufficient evidence"
        ),
    }

    return labels.get(
        difference_type,
        difference_type.replace(
            "_",
            " ",
        ).title(),
    )


def _build_title(
    disagreement: Disagreement,
    *,
    index: int | None,
) -> str:
    """
    Construct the display title for a difference.
    """
    topic = getattr(
        disagreement,
        "topic",
        None,
    )

    if not topic:
        topic = "Cross-expert difference"

    if index is not None:
        return f"{index}. {topic}"

    return str(topic)


def _format_timestamp_range(
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> str:
    """
    Format a timestamp range safely.
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
    Explain how cross-expert differences are grounded.
    """
    st.info(
        """
        **Important:** a difference is not automatically a disagreement.

        The analysis distinguishes between direct disagreement, different
        emphasis, different experiences, complementary views, and
        insufficient evidence. Each finding is backed by transcript
        evidence so the classification can be checked against the
        underlying interviews.
        """
    )


def _safe_error_message(
    error: Exception,
) -> str:
    """
    Return a user-safe exception message.
    """
    message = str(error).strip()

    if message:
        return message

    return (
        "Difference analysis could not be completed. "
        "Check the application logs for details."
    )


__all__ = [
    "render_disagreement",
    "render_disagreement_summary",
    "render_disagreement_type_guide",
    "render_disagreements",
    "render_disagreements_list",
]