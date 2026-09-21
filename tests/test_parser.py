"""
Unit tests for transcript parsing.

These tests verify that the ingestion parser correctly extracts:

- transcript metadata;
- expert information;
- speaker information;
- timestamps;
- segment text;
- deterministic document identifiers;
- segment ordering;
- duration information.

The tests use small in-memory transcript samples so they remain
fast and independent of the filesystem.
"""

from __future__ import annotations

import pytest

from exception import SensorException
from src.ingestion.parser import TranscriptParser
from src.models import SpeakerType


@pytest.fixture
def parser() -> TranscriptParser:
    """Create a parser instance for the test suite."""
    return TranscriptParser()


@pytest.fixture
def france_transcript() -> str:
    """Return a representative transcript sample."""
    return """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert: Adoption is growing, but it is concentrated in larger academic and private centres.

01:20
Expert: The biggest barrier is capital budget approval. Hospitals need a clear economic case.

02:18
Expert: ROI is very important. Finance wants utilisation, procedure volume, maintenance cost, and payback.

03:10
Expert: Training matters especially in the first year.

06:08
Expert: Six to twelve months is realistic once the hospital is serious about purchasing.
"""


def test_parser_extracts_expert_metadata(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """The parser should extract expert name, role, and market."""
    metadata, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert metadata.expert_name == "Dr. Jean Martin"
    assert metadata.expert_role == "Head of Urology"
    assert metadata.market == "France"
    assert metadata.source_file == "Transcript_1_France.txt"

    assert len(segments) > 0


def test_parser_extracts_timestamps(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """The parser should convert transcript timestamps to seconds."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    first_segment = segments[0]

    assert first_segment.start_timestamp == "00:18"
    assert first_segment.start_seconds == 18.0


def test_parser_preserves_segment_text(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Parsed segment text should contain the original statement."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert "Adoption is growing" in segments[0].text
    assert "larger academic and private centres" in segments[0].text


def test_parser_extracts_multiple_segments(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Each timestamped statement should become a separate segment."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert len(segments) == 5

    timestamps = [segment.start_timestamp for segment in segments]

    assert timestamps == [
        "00:18",
        "01:20",
        "02:18",
        "03:10",
        "06:08",
    ]


def test_parser_assigns_document_id_consistently(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """The same source should receive the same deterministic document ID."""
    metadata_one, segments_one = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    metadata_two, segments_two = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert metadata_one.document_id == metadata_two.document_id
    assert segments_one[0].document_id == segments_two[0].document_id


def test_parser_assigns_same_document_id_to_all_segments(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """All segments from one transcript must share its document ID."""
    metadata, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert all(
        segment.document_id == metadata.document_id
        for segment in segments
    )


def test_parser_assigns_unique_segment_ids(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Every parsed segment should have a unique segment identifier."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    segment_ids = [segment.segment_id for segment in segments]

    assert len(segment_ids) == len(set(segment_ids))


def test_parser_classifies_expert_speaker(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """An Expert speaker should be classified as an expert."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].speaker == "Expert"
    assert segments[0].speaker_type == SpeakerType.EXPERT


def test_parser_orders_segments_by_timestamp(
    parser: TranscriptParser,
) -> None:
    """Segments should be returned in chronological order."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

03:10
Expert: Training is important.

00:18
Expert: Adoption is growing.

02:18
Expert: ROI is very important.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    timestamps = [segment.start_seconds for segment in segments]

    assert timestamps == sorted(timestamps)


def test_parser_infers_segment_end_timestamp(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """A segment should normally end at the next segment's start."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].end_timestamp == "01:20"
    assert segments[0].end_seconds == 80.0

    assert segments[1].end_timestamp == "02:18"
    assert segments[1].end_seconds == 138.0


def test_parser_last_segment_has_no_future_timestamp(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """The final segment should not invent an end timestamp."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    last_segment = segments[-1]

    assert last_segment.end_timestamp is None
    assert last_segment.end_seconds is None


def test_parser_calculates_duration(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Transcript metadata should expose the final known timestamp."""
    metadata, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert metadata.duration_seconds == segments[-1].start_seconds


def test_parser_normalizes_extra_whitespace(
    parser: TranscriptParser,
) -> None:
    """Whitespace around transcript statements should be normalized."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert:   Adoption    is   growing    rapidly.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].text == "Adoption is growing rapidly."


def test_parser_supports_hh_mm_ss_timestamp(
    parser: TranscriptParser,
) -> None:
    """The parser should support timestamps containing hours."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

01:02:03
Expert: This is a timestamped statement.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert len(segments) == 1
    assert segments[0].start_timestamp == "01:02:03"
    assert segments[0].start_seconds == pytest.approx(3723.0)


def test_parser_rejects_empty_transcript(
    parser: TranscriptParser,
) -> None:
    """An empty transcript should not silently produce a valid corpus."""
    with pytest.raises(SensorException):
        parser.parse(
            "",
            source_file="empty.txt",
        )


def test_parser_requires_source_file(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """A source filename is required for traceable evidence."""
    with pytest.raises(SensorException):
        parser.parse(
            france_transcript,
            source_file="",
        )


def test_parser_keeps_market_at_segment_level(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Each segment should retain the transcript market."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert all(segment.market == "France" for segment in segments)


def test_parser_keeps_expert_identity_at_segment_level(
    parser: TranscriptParser,
    france_transcript: str,
) -> None:
    """Each segment should retain the expert identity."""
    _, segments = parser.parse(
        france_transcript,
        source_file="Transcript_1_France.txt",
    )

    assert all(
        segment.expert_name == "Dr. Jean Martin"
        for segment in segments
    )

    assert all(
        segment.expert_role == "Head of Urology"
        for segment in segments
    )