"""
Common-theme UI for the Hasamex Expert Analysis application.

This module renders themes identified across multiple expert transcripts.

The UI delegates theme detection to ThemeAnalyzer and focuses only on:

    - User interaction
    - Theme presentation
    - Supporting evidence
    - Expert attribution
    - Timestamp/source display

No theme, quote, expert name, or timestamp is hard-coded here.
All analytical content comes from the analysis layer.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from logger import logging

from src.analysis.themes import ThemeAnalyzer
from src.models import Theme, ThemeEvidence

logger = logging.getLogger(__name__)


def render_themes(
    analyzer: ThemeAnalyzer,
) -> None:
    """
    Render the main common-themes page.

    Args:
        analyzer:
            Initialized ThemeAnalyzer instance.
    """
    st.title("🧩 Common Themes")

    st.caption(
        "Recurring themes identified from evidence across expert transcripts."
    )

    _render_grounding_notice()

    st.markdown("### Theme Analysis")

    question = st.text_input(
        "Optional focus question",
        placeholder=(
            "Example: What factors influence robotic surgery adoption?"
        ),
        help=(
            "Leave this empty to identify broad recurring themes. "
            "Enter a question to focus theme analysis on a specific topic."
        ),
        key="theme_focus_question",
    )

    if st.button(
        "Analyze Common Themes",
        type="primary",
        use_container_width=True,
        key="analyze_themes",
    ):
        _run_theme_analysis(
            analyzer=analyzer,
            question=question.strip() or None,
        )

    themes = st.session_state.get(
        "hasamex_themes"
    )

    if themes is not None:
        st.divider()
        render_theme_results(
            themes
        )


def render_theme_results(
    themes: list[Theme],
) -> None:
    """
    Render a collection of identified themes.
    """
    if not themes:
        st.info(
            "No sufficiently supported cross-expert themes were identified."
        )
        return

    st.success(
        f"{len(themes)} supported theme(s) identified."
    )

    for index, theme in enumerate(
        themes,
        start=1,
    ):
        _render_theme(
            theme,
            index=index,
        )


def render_theme(
    theme: Theme,
    *,
    index: int | None = None,
) -> None:
    """
    Public helper for rendering one theme.
    """
    _render_theme(
        theme,
        index=index,
    )


def _render_theme(
    theme: Theme,
    *,
    index: int | None = None,
) -> None:
    """
    Render a single theme and its supporting evidence.
    """
    title = _theme_title(
        theme,
        index=index,
    )

    with st.container(border=True):
        st.markdown(
            f"### {title}"
        )

        if theme.summary:
            st.markdown(
                theme.summary
            )

        experts = _extract_experts(
            theme
        )

        if experts:
            st.markdown("**Experts represented**")

            st.write(
                " • ".join(experts)
            )

        evidence_items = _get_theme_evidence(
            theme
        )

        if evidence_items:
            st.markdown(
                "#### Supporting Evidence"
            )

            for evidence_index, evidence in enumerate(
                evidence_items,
                start=1,
            ):
                _render_theme_evidence(
                    evidence,
                    index=evidence_index,
                )
        else:
            st.warning(
                "No supporting evidence is available for this theme."
            )

        _render_theme_metadata(
            theme
        )


def _run_theme_analysis(
    analyzer: ThemeAnalyzer,
    question: str | None,
) -> None:
    """
    Execute theme analysis and store the result in Streamlit state.
    """
    try:
        with st.spinner(
            "Retrieving cross-expert evidence and identifying themes..."
        ):
            if question:
                themes = analyzer.analyze_for_question(
                    question
                )
            else:
                themes = analyzer.analyze()

        st.session_state[
            "hasamex_themes"
        ] = themes

        if themes:
            st.success(
                "Theme analysis completed."
            )
        else:
            st.info(
                "The available evidence did not support any sufficiently "
                "cross-expert themes."
            )

    except Exception as error:
        logger.exception(
            "Theme analysis failed."
        )

        st.session_state[
            "hasamex_themes"
        ] = None

        st.error(
            _safe_error_message(error)
        )


def _render_theme_evidence(
    evidence: ThemeEvidence,
    *,
    index: int,
) -> None:
    """
    Render one theme-supporting evidence item.
    """
    timestamp = _format_timestamp_range(
        evidence.start_timestamp,
        evidence.end_timestamp,
    )

    expert = (
        evidence.expert_name
        or "Unknown expert"
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

        st.markdown(
            f"**Timestamp:** `{timestamp}`"
        )

        if evidence.quote:
            st.markdown(
                f'> "{evidence.quote}"'
            )

        if evidence.citation:
            _render_citation(
                evidence.citation
            )

        if evidence.evidence_id:
            st.caption(
                f"Evidence ID: {evidence.evidence_id}"
            )


def _render_citation(
    citation: Any,
) -> None:
    """
    Render citation information without assuming a particular
    citation object implementation beyond its expected attributes.
    """
    source_file = getattr(
        citation,
        "source_file",
        None,
    )

    label = getattr(
        citation,
        "label",
        None,
    )

    if label:
        st.markdown(
            f"**Source:** `{label}`"
        )
    elif source_file:
        st.markdown(
            f"**Source:** `{source_file}`"
        )

    citation_timestamp = _format_timestamp_range(
        getattr(
            citation,
            "start_timestamp",
            None,
        ),
        getattr(
            citation,
            "end_timestamp",
            None,
        ),
    )

    if citation_timestamp != "Timestamp unavailable":
        st.caption(
            f"Source timestamp: {citation_timestamp}"
        )


def _render_theme_metadata(
    theme: Theme,
) -> None:
    """
    Render non-analytical theme metadata.
    """
    theme_id = getattr(
        theme,
        "theme_id",
        None,
    )

    if theme_id:
        st.caption(
            f"Theme ID: {theme_id}"
        )


def _extract_experts(
    theme: Theme,
) -> list[str]:
    """
    Extract unique expert names represented by the theme evidence.
    """
    experts: set[str] = set()

    direct_experts = getattr(
        theme,
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

    evidence_items = _get_theme_evidence(
        theme
    )

    for evidence in evidence_items:
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


def _get_theme_evidence(
    theme: Theme,
) -> list[ThemeEvidence]:
    """
    Safely retrieve theme evidence.
    """
    evidence = getattr(
        theme,
        "evidence",
        None,
    )

    if not evidence:
        return []

    return list(evidence)


def _theme_title(
    theme: Theme,
    *,
    index: int | None,
) -> str:
    """
    Construct the display title for a theme.
    """
    theme_name = getattr(
        theme,
        "theme",
        None,
    )

    if not theme_name:
        theme_name = "Unnamed Theme"

    if index is not None:
        return f"{index}. {theme_name}"

    return str(theme_name)


def _format_timestamp_range(
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> str:
    """
    Format an optional timestamp range.
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
    Explain how themes are grounded.
    """
    st.info(
        """
        **How themes are generated**

        Themes are derived from retrieved evidence across the available
        expert transcripts. A theme is displayed only when the analysis
        pipeline has supporting evidence from multiple experts.

        Each supporting item retains its expert attribution and
        transcript timestamp so the finding can be checked against
        the underlying source.
        """
    )


def _safe_error_message(
    error: Exception,
) -> str:
    """
    Return a user-safe error message.

    Detailed stack traces are recorded in the application log.
    """
    message = str(error).strip()

    if message:
        return message

    return (
        "Theme analysis could not be completed. "
        "Check the application logs for details."
    )


__all__ = [
    "render_theme",
    "render_theme_results",
    "render_themes",
]