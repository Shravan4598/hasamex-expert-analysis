"""
Unit tests for transcript timestamp handling.

These tests verify that timestamp parsing remains deterministic and
that transcript segments preserve correct temporal information.

The test suite covers:

- MM:SS timestamps;
- HH:MM:SS timestamps;
- conversion to seconds;
- timestamp formatting;
- chronological ordering;
- segment start/end boundaries;
- invalid timestamp handling.
"""

from __future__ import annotations

import pytest

from exception import SensorException
from src.ingestion.parser import TranscriptParser


@pytest.fixture
def parser() -> TranscriptParser:
    """Create a transcript parser for timestamp tests."""
    return TranscriptParser()


def test_mm_ss_timestamp_is_parsed_correctly(
    parser: TranscriptParser,
) -> None:
    """MM:SS timestamps should be converted to seconds."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

01:20
Expert: Capital budget approval is the biggest barrier.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert len(segments) == 1
    assert segments[0].start_timestamp == "01:20"
    assert segments[0].start_seconds == pytest.approx(80.0)


def test_hh_mm_ss_timestamp_is_parsed_correctly(
    parser: TranscriptParser,
) -> None:
    """HH:MM:SS timestamps should be converted to seconds."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

01:02:03
Expert: This is a long-form timestamp.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert len(segments) == 1
    assert segments[0].start_timestamp == "01:02:03"
    assert segments[0].start_seconds == pytest.approx(3723.0)


def test_zero_timestamp_is_supported(
    parser: TranscriptParser,
) -> None:
    """00:00 should represent the beginning of a transcript."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:00
Expert: This is the beginning of the call.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].start_timestamp == "00:00"
    assert segments[0].start_seconds == pytest.approx(0.0)


def test_timestamp_order_is_preserved_chronologically(
    parser: TranscriptParser,
) -> None:
    """Parsed segments should be ordered from earliest to latest."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

04:08
Expert: Outcomes are necessary.

00:18
Expert: Adoption is growing.

02:18
Expert: ROI is very important.

01:20
Expert: Capital budget approval is important.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert [segment.start_seconds for segment in segments] == [
        18.0,
        80.0,
        138.0,
        248.0,
    ]


def test_end_timestamp_uses_next_segment_start(
    parser: TranscriptParser,
) -> None:
    """A segment should end where the next segment begins."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert: Adoption is growing.

01:20
Expert: Capital budget approval is the biggest barrier.

02:18
Expert: ROI is very important.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].start_seconds == pytest.approx(18.0)
    assert segments[0].end_seconds == pytest.approx(80.0)

    assert segments[1].start_seconds == pytest.approx(80.0)
    assert segments[1].end_seconds == pytest.approx(138.0)


def test_last_segment_does_not_invent_end_time(
    parser: TranscriptParser,
) -> None:
    """The final segment should not receive a fabricated end time."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

06:08
Expert: Six to twelve months is realistic.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[-1].start_seconds == pytest.approx(368.0)
    assert segments[-1].end_timestamp is None
    assert segments[-1].end_seconds is None


def test_known_france_transcript_timestamps(
    parser: TranscriptParser,
) -> None:
    """Known timestamps from the France transcript should remain accurate."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert: Adoption is growing.

01:20
Expert: The biggest barrier is capital budget approval.

02:18
Expert: ROI is very important.

03:10
Expert: Training matters especially in the first year.

04:08
Expert: Outcomes are necessary but not enough.

05:07
Expert: Continued increase is expected.

06:08
Expert: Six to twelve months is realistic.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    expected = [
        ("00:18", 18.0),
        ("01:20", 80.0),
        ("02:18", 138.0),
        ("03:10", 190.0),
        ("04:08", 248.0),
        ("05:07", 307.0),
        ("06:08", 368.0),
    ]

    actual = [
        (segment.start_timestamp, segment.start_seconds)
        for segment in segments
    ]

    assert actual == expected


def test_timestamp_gap_is_calculated_from_segment_boundaries(
    parser: TranscriptParser,
) -> None:
    """The temporal gap between adjacent segments should be deterministic."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

01:20
Expert: Capital budget approval is important.

02:18
Expert: ROI is very important.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    gap = segments[1].start_seconds - segments[0].start_seconds

    assert gap == pytest.approx(58.0)


def test_timestamp_values_are_numeric(
    parser: TranscriptParser,
) -> None:
    """Start timestamps should be stored as numeric seconds."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

02:18
Expert: ROI is very important.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert isinstance(segments[0].start_seconds, float)


def test_timestamp_metadata_duration_matches_final_timestamp(
    parser: TranscriptParser,
) -> None:
    """Transcript duration should match the final known timestamp."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert: Adoption is growing.

06:08
Expert: Six to twelve months is realistic.
"""

    metadata, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert metadata.duration_seconds == segments[-1].start_seconds
    assert metadata.duration_seconds == pytest.approx(368.0)


def test_timestamp_with_single_digit_minute_is_supported(
    parser: TranscriptParser,
) -> None:
    """A normal MM:SS timestamp should support minute values below ten."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

05:07
Expert: Adoption is expected to increase.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].start_timestamp == "05:07"
    assert segments[0].start_seconds == pytest.approx(307.0)


def test_timestamp_with_large_minute_value_is_supported(
    parser: TranscriptParser,
) -> None:
    """MM:SS timestamps should support calls longer than one hour."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

65:30
Expert: This statement occurs later in the call.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert segments[0].start_seconds == pytest.approx(3930.0)


def test_invalid_timestamp_is_not_treated_as_valid_segment(
    parser: TranscriptParser,
) -> None:
    """Malformed timestamp text should not become transcript timing metadata."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

not-a-timestamp
Expert: This line does not contain a valid timestamp.
"""

    with pytest.raises(SensorException):
        parser.parse(
            transcript,
            source_file="Transcript_1_France.txt",
        )


def test_timestamps_remain_attached_to_correct_content(
    parser: TranscriptParser,
) -> None:
    """Timestamped content must remain associated with its original time."""
    transcript = """Expert 1 – Dr. Jean Martin, Head of Urology, France

00:18
Expert: Adoption is growing.

02:18
Expert: ROI is very important.

06:08
Expert: Six to twelve months is realistic.
"""

    _, segments = parser.parse(
        transcript,
        source_file="Transcript_1_France.txt",
    )

    assert "Adoption is growing" in segments[0].text
    assert segments[0].start_timestamp == "00:18"

    assert "ROI is very important" in segments[1].text
    assert segments[1].start_timestamp == "02:18"

    assert "Six to twelve months" in segments[2].text
    assert segments[2].start_timestamp == "06:08"