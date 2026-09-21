"""
Main dashboard UI for the Hasamex Expert Analysis application.

The dashboard provides:

    - Application overview
    - Corpus statistics
    - Navigation between analysis workflows
    - System status
    - Evidence-grounding explanation

The dashboard does not perform retrieval or LLM calls directly.
Those responsibilities belong to the application/service layers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from logger import logging
from src.models import ApplicationStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardNavigation:
    """
    Navigation configuration for the Streamlit application.
    """

    pages: tuple[str, ...] = (
        "Dashboard",
        "Interview Guide",
        "Themes",
        "Disagreements",
        "Ask Questions",
        "Sources",
    )


def render_dashboard(
    stats: ApplicationStats | None = None,
    *,
    navigation: DashboardNavigation | None = None,
) -> str:
    """
    Render the main application dashboard.

    Args:
        stats:
            Runtime application statistics. If unavailable, the UI
            displays a safe empty state rather than fabricated values.

        navigation:
            Optional navigation configuration.

    Returns:
        Name of the currently selected page.
    """
    navigation = navigation or DashboardNavigation()

    _configure_page()
    _render_header()
    _render_navigation(navigation)

    selected_page = st.session_state.get(
        "selected_page",
        navigation.pages[0],
    )

    if selected_page == "Dashboard":
        _render_overview(stats)
    else:
        _render_navigation_hint(selected_page)

    return selected_page


def render_sidebar(
    stats: ApplicationStats | None = None,
    *,
    navigation: DashboardNavigation | None = None,
) -> str:
    """
    Render the sidebar navigation and corpus status.

    This helper can be used by the application entry point when each
    page is rendered by a separate UI module.
    """
    navigation = navigation or DashboardNavigation()

    with st.sidebar:
        st.markdown(
            "## 🔎 Hasamex Expert Analysis"
        )

        st.caption(
            "Source-grounded analysis of expert interview transcripts."
        )

        selected_page = st.radio(
            "Navigate",
            options=navigation.pages,
            index=_safe_navigation_index(
                st.session_state.get(
                    "selected_page",
                    navigation.pages[0],
                ),
                navigation.pages,
            ),
            key="dashboard_navigation",
        )

        st.session_state["selected_page"] = selected_page

        st.divider()

        st.markdown("### Corpus")

        if stats is None:
            st.info(
                "No transcript corpus is loaded yet."
            )
        else:
            _render_sidebar_stats(stats)

        st.divider()

        st.caption(
            "Answers are generated only from retrieved transcript "
            "evidence. Unsupported claims are refused."
        )

    return selected_page


def render_header(
    title: str = "Hasamex Expert Analysis",
    subtitle: str = (
        "Evidence-grounded analysis across expert interview transcripts"
    ),
) -> None:
    """
    Render a reusable application header.
    """
    st.markdown(
        f"""
        <div style="
            padding: 0.5rem 0 1rem 0;
        ">
            <h1 style="margin-bottom: 0.25rem;">
                {title}
            </h1>
            <p style="
                color: #666;
                font-size: 1.05rem;
                margin-top: 0;
            ">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats_cards(
    stats: ApplicationStats | None,
) -> None:
    """
    Render corpus statistics as metric cards.
    """
    if stats is None:
        st.warning(
            "Corpus statistics are not available."
        )
        return

    columns = st.columns(4)

    metric_values = (
        (
            "Experts",
            stats.expert_count,
            "Unique experts in the loaded corpus",
        ),
        (
            "Transcripts",
            stats.transcript_count,
            "Loaded transcript documents",
        ),
        (
            "Segments",
            stats.segment_count,
            "Parsed timestamped segments",
        ),
        (
            "Chunks",
            stats.chunk_count,
            "Retrieval-ready chunks",
        ),
    )

    for column, (label, value, help_text) in zip(
        columns,
        metric_values,
    ):
        with column:
            st.metric(
                label=label,
                value=value,
                help=help_text,
            )


def render_grounding_notice() -> None:
    """
    Explain the application's grounding strategy.
    """
    st.info(
        """
        **Grounding policy**

        The application retrieves relevant transcript evidence before
        generating an answer. Expert names, timestamps, source files,
        and citations are taken from the source data rather than
        generated by the language model.

        If sufficient evidence cannot be found, the application
        returns an evidence-limited response instead of guessing.
        """
    )


def render_system_status(
    *,
    corpus_loaded: bool,
    vector_store_ready: bool,
    llm_ready: bool,
) -> None:
    """
    Render the current system readiness state.
    """
    st.subheader("System Status")

    columns = st.columns(3)

    status_items = (
        (
            "Transcript Corpus",
            corpus_loaded,
        ),
        (
            "Vector Store",
            vector_store_ready,
        ),
        (
            "LLM Service",
            llm_ready,
        ),
    )

    for column, (label, ready) in zip(
        columns,
        status_items,
    ):
        with column:
            if ready:
                st.success(
                    f"{label}: Ready"
                )
            else:
                st.warning(
                    f"{label}: Not ready"
                )


def render_empty_corpus_state() -> None:
    """
    Render a useful empty state when no transcripts are loaded.
    """
    st.warning(
        "No transcript corpus is currently available."
    )

    st.markdown(
        """
        ### To get started

        1. Place the transcript files in the configured raw-data
           directory, or use the application's ingestion workflow.
        2. Parse the transcripts.
        3. Build or load the vector index.
        4. Return to the analysis pages.

        Supported transcript formats are plain-text and Markdown
        files.
        """
    )


def _configure_page() -> None:
    """
    Apply safe Streamlit page configuration.

    This function is intentionally defensive because page
    configuration can only be called once per Streamlit execution.
    """
    try:
        st.set_page_config(
            page_title="Hasamex Expert Analysis",
            page_icon="🔎",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception as error:
        logger.debug(
            "Streamlit page configuration was already initialized: %s",
            error,
        )


def _render_header() -> None:
    """
    Render the dashboard header.
    """
    render_header()


def _render_navigation(
    navigation: DashboardNavigation,
) -> None:
    """
    Render top-level navigation controls.
    """
    selected_page = st.session_state.get(
        "selected_page",
        navigation.pages[0],
    )

    columns = st.columns(
        len(navigation.pages)
    )

    for column, page in zip(
        columns,
        navigation.pages,
    ):
        with column:
            button_type = (
                "primary"
                if page == selected_page
                else "secondary"
            )

            if st.button(
                page,
                key=f"dashboard_nav_{page}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state[
                    "selected_page"
                ] = page
                st.rerun()


def _render_overview(
    stats: ApplicationStats | None,
) -> None:
    """
    Render the dashboard overview page.
    """
    st.subheader(
        "Expert Intelligence Workspace"
    )

    st.markdown(
        """
        Analyze expert interviews using a source-grounded RAG
        workflow. The application supports structured interview-guide
        answers, cross-expert themes, differences, and free-form
        questions.
        """
    )

    render_grounding_notice()

    st.markdown("### Corpus Overview")

    if stats is None:
        render_empty_corpus_state()
        return

    render_stats_cards(stats)

    st.markdown("### Available Analysis")

    analysis_columns = st.columns(4)

    analysis_cards = (
        (
            analysis_columns[0],
            "📋 Interview Guide",
            "Answer the predefined interview questions for the "
            "available experts.",
            "Interview Guide",
        ),
        (
            analysis_columns[1],
            "🧩 Common Themes",
            "Identify recurring topics and supporting evidence "
            "across experts.",
            "Themes",
        ),
        (
            analysis_columns[2],
            "⚖️ Differences",
            "Surface disagreements, different emphasis, and "
            "different market experiences.",
            "Disagreements",
        ),
        (
            analysis_columns[3],
            "💬 Ask Questions",
            "Ask natural-language questions across the transcript "
            "corpus.",
            "Ask Questions",
        ),
    )

    for (
        column,
        title,
        description,
        target_page,
    ) in analysis_cards:
        with column:
            st.markdown(
                f"#### {title}"
            )
            st.caption(description)

            if st.button(
                f"Open {target_page}",
                key=f"open_{target_page}",
                use_container_width=True,
            ):
                st.session_state[
                    "selected_page"
                ] = target_page
                st.rerun()

    st.markdown("### Source Traceability")

    st.markdown(
        """
        Every grounded response is designed to preserve:

        - **Expert attribution**
        - **Source transcript**
        - **Timestamp**
        - **Supporting evidence**
        - **Citation metadata**

        This makes the generated analysis auditable against the
        underlying interviews.
        """
    )


def _render_navigation_hint(
    selected_page: str,
) -> None:
    """
    Display a lightweight placeholder when a specialized page is
    rendered elsewhere by the application router.
    """
    st.info(
        f"Use the **{selected_page}** page to continue."
    )


def _render_sidebar_stats(
    stats: ApplicationStats,
) -> None:
    """
    Render compact corpus statistics in the sidebar.
    """
    st.metric(
        "Experts",
        stats.expert_count,
    )

    st.metric(
        "Transcripts",
        stats.transcript_count,
    )

    st.metric(
        "Segments",
        stats.segment_count,
    )

    st.metric(
        "Chunks",
        stats.chunk_count,
    )


def _safe_navigation_index(
    selected_page: str,
    pages: tuple[str, ...],
) -> int:
    """
    Return a valid radio-button index.
    """
    try:
        return pages.index(selected_page)
    except ValueError:
        return 0


def render_error(
    message: str,
    *,
    title: str = "Something went wrong",
) -> None:
    """
    Render a consistent application error message.
    """
    st.error(
        f"**{title}**\n\n{message}"
    )


def render_loading(
    message: str = "Processing transcript evidence...",
) -> None:
    """
    Render a consistent loading indicator.
    """
    st.spinner(message)


def render_action_button(
    label: str,
    callback: Callable[[], None],
    *,
    key: str,
    disabled: bool = False,
) -> bool:
    """
    Render a reusable action button.

    Args:
        label: Button label.
        callback: Function executed when the button is clicked.
        key: Unique Streamlit widget key.
        disabled: Whether the button is disabled.

    Returns:
        True when the button was clicked.
    """
    clicked = st.button(
        label,
        key=key,
        disabled=disabled,
    )

    if clicked:
        callback()

    return clicked


__all__ = [
    "DashboardNavigation",
    "render_action_button",
    "render_dashboard",
    "render_error",
    "render_grounding_notice",
    "render_header",
    "render_loading",
    "render_sidebar",
    "render_stats_cards",
    "render_system_status",
]