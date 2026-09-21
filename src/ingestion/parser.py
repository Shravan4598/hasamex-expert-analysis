"""
Transcript parsing and timestamp extraction.

The parser converts raw expert-call transcripts into structured,
timestamp-aware transcript segments while preserving source traceability.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import unicodedata
from typing import List, Optional, Tuple

from exception import SensorException
from src.models import (
    SpeakerType,
    TranscriptMetadata,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class TranscriptParser:
    """Parse raw expert transcripts into structured segments."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    EXPERT_HEADER_PATTERN = re.compile(
        r"""
        ^\s*
        Expert\s+
        (?P<number>\d+)
        \s*
        [-–—:]
        \s*
        (?P<details>.+?)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    EXPERT_NAME_PATTERN = re.compile(
        r"^\s*(?:Expert\s*Name|Name)\s*:\s*(?P<value>.+?)\s*$",
        re.IGNORECASE,
    )

    ROLE_PATTERN = re.compile(
        r"^\s*(?:Role|Expert\s*Role|Title)\s*:\s*(?P<value>.+?)\s*$",
        re.IGNORECASE,
    )

    MARKET_PATTERN = re.compile(
        r"^\s*(?:Market|Country|Region)\s*:\s*(?P<value>.+?)\s*$",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------

    HH_MM_SS_PATTERN = re.compile(
        r"^\s*(?P<hours>\d{1,3}):"
        r"(?P<minutes>\d{2}):"
        r"(?P<seconds>\d{2})\s*$"
    )

    MM_SS_PATTERN = re.compile(
        r"^\s*(?P<minutes>\d{1,3}):"
        r"(?P<seconds>\d{2})\s*$"
    )

    # ------------------------------------------------------------------
    # Speaker
    # ------------------------------------------------------------------

    SPEAKER_PATTERN = re.compile(
        r"^\s*"
        r"(?P<speaker>[A-Za-z][A-Za-z0-9 _-]{0,50})"
        r"\s*:\s*"
        r"(?P<text>.*)$"
    )

    def parse(
        self,
        text: str,
        source_file: str,
    ) -> Tuple[TranscriptMetadata, List[TranscriptSegment]]:
        """
        Parse a complete transcript.

        Returns
        -------
        tuple
            Transcript metadata and chronologically ordered segments.
        """

        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Transcript text cannot be empty.")

            normalized_text = self._normalize_text(text)
            lines = normalized_text.splitlines()

            _, header_match = self._find_expert_header(lines)

            if header_match is None:
                raise ValueError(
                    f"Could not find expert header in '{source_file}'."
                )

            # ----------------------------------------------------------
            # Expert metadata
            # ----------------------------------------------------------

            expert_name, expert_role, market = (
                self._extract_header_metadata(header_match)
            )

            explicit_name, explicit_role, explicit_market = (
                self._extract_explicit_metadata(lines)
            )

            expert_name = explicit_name or expert_name
            expert_role = explicit_role or expert_role
            market = explicit_market or market

            if not expert_name:
                raise ValueError(
                    f"Could not determine expert name in '{source_file}'."
                )

            if not market:
                raise ValueError(
                    f"Could not determine market in '{source_file}'."
                )

            # ----------------------------------------------------------
            # Document identity
            # ----------------------------------------------------------

            document_id = self._build_document_id(
                source_file=source_file,
                expert_name=expert_name,
                market=market,
            )

            # ----------------------------------------------------------
            # Parse timestamped segments
            # ----------------------------------------------------------

            segments = self._parse_segments(
                lines=lines,
                document_id=document_id,
                source_file=source_file,
                expert_name=expert_name,
                expert_role=expert_role,
                market=market,
            )

            if not segments:
                raise ValueError(
                    f"No timestamped transcript segments found in "
                    f"'{source_file}'."
                )

            # ----------------------------------------------------------
            # Chronological ordering
            # ----------------------------------------------------------

            segments.sort(
                key=lambda segment: (
                    float(segment.start_seconds),
                    segment.segment_id,
                )
            )

            # ----------------------------------------------------------
            # Infer end timestamps from next segment
            # ----------------------------------------------------------

            self._assign_segment_boundaries(segments)

            duration_seconds = float(
                segments[-1].start_seconds
            )

            metadata = TranscriptMetadata(
                document_id=document_id,
                source_file=source_file,
                expert_name=expert_name,
                expert_role=expert_role,
                market=market,
                duration_seconds=duration_seconds,
                segment_count=len(segments),
            )

            return metadata, segments

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to parse transcript '%s'.",
                source_file,
            )

            raise SensorException(
                str(error),
                sys,
            ) from error

    # ==================================================================
    # Normalization
    # ==================================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize line endings and Unicode punctuation.

        Transcript wording is preserved, while formatting whitespace
        is normalized.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = unicodedata.normalize("NFKC", text)

        # Normalize dash variants.
        for dash in (
            "\u2010",
            "\u2011",
            "\u2012",
            "\u2013",
            "\u2014",
            "\u2212",
        ):
            text = text.replace(dash, "-")

        # Normalize non-breaking spaces.
        text = text.replace("\u00a0", " ")

        return text

    @staticmethod
    def _normalize_segment_text(text: str) -> str:
        """Collapse repeated whitespace inside segment content."""

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    # ==================================================================
    # Header parsing
    # ==================================================================

    def _find_expert_header(
        self,
        lines: List[str],
    ) -> Tuple[int, Optional[re.Match[str]]]:
        """Find the expert header."""

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()

            if not line:
                continue

            match = self.EXPERT_HEADER_PATTERN.match(line)

            if match:
                return index, match

        return -1, None

    @staticmethod
    def _extract_header_metadata(
        header_match: re.Match[str],
    ) -> Tuple[str, str, str]:
        """
        Extract metadata from compact header.

        Example:
            Expert 1 - Dr. Jean Martin, Head of Urology, France
        """

        details = header_match.group("details").strip()

        parts = [
            part.strip()
            for part in details.split(",")
            if part.strip()
        ]

        if not parts:
            return "", "", ""

        expert_name = parts[0]

        if len(parts) >= 3:
            expert_role = ", ".join(parts[1:-1])
            market = parts[-1]

        elif len(parts) == 2:
            expert_role = ""
            market = parts[-1]

        else:
            expert_role = ""
            market = ""

        return (
            expert_name,
            expert_role,
            market,
        )

    def _extract_explicit_metadata(
        self,
        lines: List[str],
    ) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        """Extract optional explicit metadata fields."""

        expert_name: Optional[str] = None
        expert_role: Optional[str] = None
        market: Optional[str] = None

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                continue

            name_match = self.EXPERT_NAME_PATTERN.match(line)

            if name_match:
                expert_name = (
                    name_match.group("value").strip()
                )
                continue

            role_match = self.ROLE_PATTERN.match(line)

            if role_match:
                expert_role = (
                    role_match.group("value").strip()
                )
                continue

            market_match = self.MARKET_PATTERN.match(line)

            if market_match:
                market = (
                    market_match.group("value").strip()
                )

        return (
            expert_name,
            expert_role,
            market,
        )

    # ==================================================================
    # Segment parsing
    # ==================================================================

    def _parse_segments(
        self,
        lines: List[str],
        document_id: str,
        source_file: str,
        expert_name: str,
        expert_role: str,
        market: str,
    ) -> List[TranscriptSegment]:
        """
        Parse timestamped transcript blocks.

        Timestamps are collected first and chronological ordering is
        performed after the complete transcript has been parsed.
        """

        raw_segments: List[dict] = []

        current_timestamp: Optional[str] = None
        current_seconds: Optional[float] = None
        current_text_lines: List[str] = []
        current_speaker: str = "Expert"

        def flush_segment() -> None:
            nonlocal current_timestamp
            nonlocal current_seconds
            nonlocal current_text_lines
            nonlocal current_speaker

            if (
                current_timestamp is None
                or current_seconds is None
            ):
                return

            text_value = self._normalize_segment_text(
                " ".join(current_text_lines)
            )

            if not text_value:
                logger.warning(
                    "Skipping timestamp %s in %s because no content "
                    "was found.",
                    current_timestamp,
                    source_file,
                )

                current_timestamp = None
                current_seconds = None
                current_text_lines = []
                current_speaker = "Expert"
                return

            raw_segments.append(
                {
                    "start_timestamp": current_timestamp,
                    "start_seconds": float(current_seconds),
                    "speaker": current_speaker,
                    "text": text_value,
                }
            )

            current_timestamp = None
            current_seconds = None
            current_text_lines = []
            current_speaker = "Expert"

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                continue

            # ----------------------------------------------------------
            # Metadata/header lines
            # ----------------------------------------------------------

            if self.EXPERT_HEADER_PATTERN.match(line):
                continue

            if self.EXPERT_NAME_PATTERN.match(line):
                continue

            if self.ROLE_PATTERN.match(line):
                continue

            if self.MARKET_PATTERN.match(line):
                continue

            # ----------------------------------------------------------
            # Timestamp
            # ----------------------------------------------------------

            timestamp_seconds = self._parse_timestamp(line)

            if timestamp_seconds is not None:
                flush_segment()

                current_timestamp = line
                current_seconds = float(timestamp_seconds)
                current_text_lines = []
                current_speaker = "Expert"

                continue

            # ----------------------------------------------------------
            # Speaker line
            # ----------------------------------------------------------

            speaker_match = self.SPEAKER_PATTERN.match(line)

            if speaker_match:
                detected_speaker = (
                    speaker_match.group("speaker").strip()
                )

                content = (
                    speaker_match.group("text").strip()
                )

                if current_timestamp is not None:
                    current_speaker = (
                        detected_speaker or "Expert"
                    )

                    if content:
                        current_text_lines.append(content)

                    continue

            # ----------------------------------------------------------
            # Continuation line
            # ----------------------------------------------------------

            if current_timestamp is not None:
                current_text_lines.append(line)

        # Flush the final timestamp block.
        flush_segment()

        # --------------------------------------------------------------
        # Create Pydantic segments.
        #
        # IDs are initially assigned according to chronological order.
        # --------------------------------------------------------------

        raw_segments.sort(
            key=lambda item: float(
                item["start_seconds"]
            )
        )

        segments: List[TranscriptSegment] = []

        for index, item in enumerate(
            raw_segments,
            start=1,
        ):
            segment_id = (
                f"{document_id}-segment-{index:04d}"
            )

            speaker = item["speaker"]

            segments.append(
                TranscriptSegment(
                    segment_id=segment_id,
                    document_id=document_id,
                    source_file=source_file,

                    # Segment-level identity.
                    expert_name=expert_name,
                    expert_role=expert_role,
                    market=market,

                    speaker=speaker,
                    speaker_type=self._classify_speaker(
                        speaker
                    ),

                    text=item["text"],

                    start_timestamp=item[
                        "start_timestamp"
                    ],
                    end_timestamp=None,

                    start_seconds=float(
                        item["start_seconds"]
                    ),
                    end_seconds=None,
                )
            )

        return segments

    # ==================================================================
    # Segment boundaries
    # ==================================================================

    @staticmethod
    def _assign_segment_boundaries(
        segments: List[TranscriptSegment],
    ) -> None:
        """
        Set each segment's end timestamp to the next segment's start.

        The final segment intentionally has no invented end timestamp.
        """

        for index, segment in enumerate(segments):
            if index + 1 >= len(segments):
                segment.end_timestamp = None
                segment.end_seconds = None
                continue

            next_segment = segments[index + 1]

            segment.end_timestamp = (
                next_segment.start_timestamp
            )

            segment.end_seconds = float(
                next_segment.start_seconds
            )

    # ==================================================================
    # Timestamp parsing
    # ==================================================================

    def _parse_timestamp(
        self,
        value: str,
    ) -> Optional[float]:
        """Convert MM:SS or HH:MM:SS into seconds."""

        value = value.strip()

        # HH:MM:SS
        match = self.HH_MM_SS_PATTERN.match(value)

        if match:
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))

            if minutes >= 60 or seconds >= 60:
                return None

            return float(
                hours * 3600
                + minutes * 60
                + seconds
            )

        # MM:SS
        match = self.MM_SS_PATTERN.match(value)

        if match:
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))

            if seconds >= 60:
                return None

            return float(
                minutes * 60
                + seconds
            )

        return None

    # ==================================================================
    # Speaker classification
    # ==================================================================

    @staticmethod
    def _classify_speaker(
        speaker: str,
    ) -> SpeakerType:
        """Classify transcript speaker."""

        normalized = speaker.strip().lower()

        if normalized == "expert":
            return SpeakerType.EXPERT

        if normalized.startswith("expert "):
            return SpeakerType.EXPERT

        if normalized == "interviewer":
            return SpeakerType.INTERVIEWER

        if normalized == "moderator":
            return SpeakerType.MODERATOR

        return SpeakerType.UNKNOWN

    # ==================================================================
    # Document ID
    # ==================================================================

    @staticmethod
    def _build_document_id(
        source_file: str,
        expert_name: str,
        market: str,
    ) -> str:
        """Create deterministic document identifier."""

        source_name = source_file.rsplit(
            "/",
            1,
        )[-1]

        source_name = source_name.rsplit(
            "\\",
            1,
        )[-1]

        if "." in source_name:
            source_name = source_name.rsplit(
                ".",
                1,
            )[0]

        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            source_name.lower(),
        ).strip("-")

        identity = (
            f"{source_file}|"
            f"{expert_name}|"
            f"{market}"
        )

        digest = hashlib.sha1(
            identity.encode("utf-8")
        ).hexdigest()[:8]

        return f"{slug}-{digest}"