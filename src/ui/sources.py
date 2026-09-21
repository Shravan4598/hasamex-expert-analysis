"""
Source explorer UI for the Hasamex Expert Analysis application.

This module provides a transparent view of the transcript corpus.

The source explorer is intentionally read-only. It does not generate
answers or perform independent reasoning. Its purpose is to let users
inspect the underlying material used by the RAG and analysis pipelines.

The UI supports:

    - Transcript/source selection
    - Expert and market metadata
    - Timestamped transcript segments
    - Source text inspection
    - Evidence traceability
    - Basic corpus statistics

All displayed content comes from the loaded transcript/retrieval
objects. No transcript content is hard-coded.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import streamlit as st

from logger import logging

from src.models import (
    TranscriptChunk,
    TranscriptMetadata,
    TranscriptSegment,
)
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


def render_sources(
    retriever: Retriever,
    *,
    metadata: list[TranscriptMetadata] | None = None,
) -> None:
    """
    Render the source explorer page.

    Args:
        retriever:
            Initialized application retriever.

        metadata:
            Optional transcript metadata collection. When supplied,
            it is used for richer source-level information.
    """
    st.title("📚 Sources")

    st.caption(
        "Explore the transcript evidence behind the analysis."
    )

    _render_traceability_notice()

    try:
        chunks = _load_source_chunks(
            retriever
        )
    except Exception as error:
        logger.exception(
            "Failed to load source chunks."
        )

        st.error(
            _safe_error_message(error)
        )
        return

    if not chunks:
        _render_empty_sources()
        return

    st.markdown("### Corpus Overview")

    _render_source_statistics(
        chunks
    )

    st.divider()

    source_names = _extract_source_names(
        chunks
    )

    selected_source = _render_source_selector(
        source_names
    )

    filtered_chunks = _filter_chunks_by_source(
        chunks,
        selected_source,
    )

    _render_source_details(
        filtered_chunks,
        metadata=metadata,
    )

    st.divider()

    _render_transcript_explorer(
        filtered_chunks
    )


def render_source_chunks(
    chunks: list[TranscriptChunk],
) -> None:
    """
    Render a list of transcript chunks.
    """
    if not chunks:
        st.info(
            "No transcript chunks are available."
        )
        return

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        _render_chunk(
            chunk,
            index=index,
        )


def render_source_metadata(
    metadata: TranscriptMetadata,
) -> None:
    """
    Render metadata for one transcript.
    """
    st.markdown(
        f"### {metadata.source_file}"
    )

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Expert",
            metadata.expert_name or "Unknown",
        )

    with columns[1]:
        st.metric(
            "Market",
            metadata.market or "Unknown",
        )

    with columns[2]:
        st.metric(
            "Segments",
            metadata.segment_count,
        )

    with columns[3]:
        st.metric(
            "Duration",
            _format_duration(
                metadata.duration_seconds
            ),
        )

    if metadata.expert_role:
        st.caption(
            f"Role: {metadata.expert_role}"
        )

    if metadata.document_id:
        st.caption(
            f"Document ID: {metadata.document_id}"
        )


def render_single_source(
    chunks: list[TranscriptChunk],
    source_file: str,
) -> None:
    """
    Render one selected source independently.
    """
    source_chunks = _filter_chunks_by_source(
        chunks,
        source_file,
    )

    if not source_chunks:
        st.info(
            "No chunks were found for this source."
        )
        return

    _render_source_details(
        source_chunks
    )

    _render_transcript_explorer(
        source_chunks
    )


def _load_source_chunks(
    retriever: Retriever,
) -> list[TranscriptChunk]:
    """
    Load all available chunks from the retrieval layer.

    The Retriever owns access to the vector store. The source explorer
    intentionally uses the retrieval abstraction rather than reading
    vector-store internals directly.
    """
    chunks = retriever.get_source_chunks()

    if not chunks:
        return []

    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.source_file or "",
            chunk.start_seconds
            if chunk.start_seconds is not None
            else float("inf"),
            chunk.chunk_id,
        ),
    )


def _extract_source_names(
    chunks: Iterable[TranscriptChunk],
) -> list[str]:
    """
    Extract unique source-file names.
    """
    names = {
        chunk.source_file.strip()
        for chunk in chunks
        if chunk.source_file
        and chunk.source_file.strip()
    }

    return sorted(
        names,
        key=str.casefold,
    )


def _render_source_selector(
    source_names: list[str],
) -> str:
    """
    Render the source selection control.
    """
    options = [
        "All sources",
        *source_names,
    ]

    selected = st.selectbox(
        "Transcript source",
        options=options,
        key="source_explorer_source",
    )

    if selected == "All sources":
        return ""

    return selected


def _filter_chunks_by_source(
    chunks: list[TranscriptChunk],
    source_file: str,
) -> list[TranscriptChunk]:
    """
    Filter transcript chunks by source file.
    """
    if not source_file:
        return list(chunks)

    normalized_source = source_file.casefold()

    return [
        chunk
        for chunk in chunks
        if (
            chunk.source_file
            and chunk.source_file.casefold()
            == normalized_source
        )
    ]


def _render_source_details(
    chunks: list[TranscriptChunk],
    *,
    metadata: list[TranscriptMetadata] | None = None,
) -> None:
    """
    Render source-level metadata derived from selected chunks.
    """
    if not chunks:
        return

    source_groups: dict[str, list[TranscriptChunk]] = (
        defaultdict(list)
    )

    for chunk in chunks:
        source_groups[
            chunk.source_file or "Unknown source"
        ].append(chunk)

    if len(source_groups) == 1:
        source_file, source_chunks = next(
            iter(source_groups.items())
        )

        matching_metadata = _find_metadata(
            metadata,
            source_file,
        )

        if matching_metadata is not None:
            render_source_metadata(
                matching_metadata
            )
            return

        _render_derived_source_metadata(
            source_file,
            source_chunks,
        )
        return

    st.markdown(
        "### Loaded Sources"
    )

    for source_file, source_chunks in source_groups.items():
        with st.expander(
            source_file,
            expanded=False,
        ):
            matching_metadata = _find_metadata(
                metadata,
                source_file,
            )

            if matching_metadata is not None:
                render_source_metadata(
                    matching_metadata
                )
            else:
                _render_derived_source_metadata(
                    source_file,
                    source_chunks,
                )


def _render_derived_source_metadata(
    source_file: str,
    chunks: list[TranscriptChunk],
) -> None:
    """
    Render metadata derived directly from retrieval chunks when the
    full TranscriptMetadata object is unavailable.
    """
    experts = _unique_values(
        chunk.expert_name
        for chunk in chunks
    )

    markets = _unique_values(
        chunk.market
        for chunk in chunks
    )

    segment_ids = {
        segment_id
        for chunk in chunks
        for segment_id in chunk.segment_ids
    }

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Chunks",
            len(chunks),
        )

    with columns[1]:
        st.metric(
            "Segments",
            len(segment_ids),
        )

    with columns[2]:
        st.metric(
            "Experts",
            len(experts),
        )

    with columns[3]:
        st.metric(
            "Markets",
            len(markets),
        )

    st.markdown(
        f"**Source:** `{source_file}`"
    )

    if experts:
        st.markdown(
            f"**Experts:** {', '.join(experts)}"
        )

    if markets:
        st.markdown(
            f"**Markets:** {', '.join(markets)}"
        )


def _render_transcript_explorer(
    chunks: list[TranscriptChunk],
) -> None:
    """
    Render the timestamp-aware transcript explorer.
    """
    st.markdown(
        "### Transcript Evidence"
    )

    if not chunks:
        st.info(
            "No transcript evidence is available."
        )
        return

    search_text = st.text_input(
        "Search within displayed transcript evidence",
        placeholder="Search for a keyword or phrase...",
        key="source_text_search",
    )

    filtered_chunks = _filter_chunks_by_text(
        chunks,
        search_text,
    )

    st.caption(
        f"Showing {len(filtered_chunks)} of "
        f"{len(chunks)} retrieval chunk(s)."
    )

    if not filtered_chunks:
        st.warning(
            "No transcript chunks match the current search."
        )
        return

    for index, chunk in enumerate(
        filtered_chunks,
        start=1,
    ):
        _render_chunk(
            chunk,
            index=index,
        )


def _render_chunk(
    chunk: TranscriptChunk,
    *,
    index: int,
) -> None:
    """
    Render one transcript chunk.
    """
    timestamp = _format_timestamp_range(
        chunk.start_timestamp,
        chunk.end_timestamp,
    )

    expert = (
        chunk.expert_name
        or "Unknown expert"
    )

    source = (
        chunk.source_file
        or "Unknown source"
    )

    label = (
        f"{index}. {expert} · {timestamp}"
    )

    with st.expander(
        label,
        expanded=False,
    ):
        columns = st.columns(3)

        with columns[0]:
            st.markdown(
                f"**Source:** `{source}`"
            )

        with columns[1]:
            st.markdown(
                f"**Timestamp:** `{timestamp}`"
            )

        with columns[2]:
            st.markdown(
                f"**Chunk ID:** `{chunk.chunk_id}`"
            )

        if chunk.market:
            st.markdown(
                f"**Market:** {chunk.market}"
            )

        if chunk.speaker:
            st.markdown(
                f"**Speaker:** {chunk.speaker}"
            )

        if chunk.segment_ids:
            st.caption(
                "Segment IDs: "
                + ", ".join(
                    chunk.segment_ids
                )
            )

        st.markdown(
            "#### Source Text"
        )

        st.text(
            chunk.text
        )


def _filter_chunks_by_text(
    chunks: list[TranscriptChunk],
    search_text: str,
) -> list[TranscriptChunk]:
    """
    Filter chunks by case-insensitive text search.
    """
    cleaned = (
        " ".join(search_text.split())
        .strip()
        .casefold()
    )

    if not cleaned:
        return chunks

    return [
        chunk
        for chunk in chunks
        if cleaned in chunk.text.casefold()
    ]


def _find_metadata(
    metadata: list[TranscriptMetadata] | None,
    source_file: str,
) -> TranscriptMetadata | None:
    """
    Find metadata belonging to a source file.
    """
    if not metadata:
        return None

    normalized = source_file.casefold()

    for item in metadata:
        if (
            item.source_file
            and item.source_file.casefold()
            == normalized
        ):
            return item

    return None


def _render_source_statistics(
    chunks: list[TranscriptChunk],
) -> None:
    """
    Render descriptive corpus statistics.
    """
    source_count = len(
        {
            chunk.source_file
            for chunk in chunks
            if chunk.source_file
        }
    )

    expert_count = len(
        {
            chunk.expert_name
            for chunk in chunks
            if chunk.expert_name
        }
    )

    market_count = len(
        {
            chunk.market
            for chunk in chunks
            if chunk.market
        }
    )

    segment_count = len(
        {
            segment_id
            for chunk in chunks
            for segment_id in chunk.segment_ids
        }
    )

    columns = st.columns(4)

    metrics = (
        (
            columns[0],
            "Sources",
            source_count,
        ),
        (
            columns[1],
            "Experts",
            expert_count,
        ),
        (
            columns[2],
            "Markets",
            market_count,
        ),
        (
            columns[3],
            "Segments",
            segment_count,
        ),
    )

    for column, label, value in metrics:
        with column:
            st.metric(
                label,
                value,
            )


def _unique_values(
    values: Iterable[str | None],
) -> list[str]:
    """
    Return normalized unique non-empty values.
    """
    result = {
        value.strip()
        for value in values
        if isinstance(
            value,
            str,
        )
        and value.strip()
    }

    return sorted(
        result,
        key=str.casefold,
    )


def _format_timestamp_range(
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> str:
    """
    Format a timestamp range.
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


def _format_duration(
    seconds: int | float | None,
) -> str:
    """
    Convert seconds into a human-readable duration.
    """
    if seconds is None:
        return "Unknown"

    try:
        total_seconds = max(
            0,
            int(seconds),
        )
    except (
        TypeError,
        ValueError,
    ):
        return "Unknown"

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, seconds_remaining = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}h "
            f"{minutes}m"
        )

    if minutes:
        return (
            f"{minutes}m "
            f"{seconds_remaining}s"
        )

    return f"{seconds_remaining}s"


def _render_traceability_notice() -> None:
    """
    Explain why the source explorer exists.
    """
    st.info(
        """
        **Source traceability**

        This page exposes the transcript material used by the analysis
        pipeline. You can inspect the source file, expert, market,
        timestamp, segment IDs, and retrieval chunk text behind the
        generated analysis.

        The source explorer is read-only and does not modify transcript
        content.
        """
    )


def _render_empty_sources() -> None:
    """
    Render an empty-state message when no source chunks exist.
    """
    st.warning(
        "No transcript sources are currently available."
    )

    st.markdown(
        """
        ### Next steps

        1. Add the transcript files to the configured raw-data
           directory.
        2. Run the ingestion pipeline.
        3. Build or load the vector store.
        4. Return to this page to inspect the indexed sources.
        """
    )


def _safe_error_message(
    error: Exception,
) -> str:
    """
    Return a UI-safe error message.
    """
    message = str(error).strip()

    if message:
        return message

    return (
        "The source explorer could not load the transcript corpus. "
        "Check the application logs for details."
    )


__all__ = [
    "render_single_source",
    "render_source_chunks",
    "render_source_metadata",
    "render_sources",
]