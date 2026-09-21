"""
Transcript parsing and timestamp extraction.

The parser converts raw expert-call transcripts into structured,
timestamp-aware TranscriptSegment objects.

Expected transcript format is similar to:

    Expert 1 – Dr. Jean Martin, Head of Urology, France.

    00:00 Interviewer: How would you describe current adoption...
    00:18 Dr. Martin: Adoption is growing...

The parser is intentionally conservative. It does not invent missing
timestamps, speaker names, expert metadata, or content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from exception import SensorException
from logger import logging

from src.models import (
    SpeakerType,
    TranscriptMetadata,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


TIMESTAMP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<hours>\d{1,2}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)"
    r"|(?P<minutes_only>[0-5]?\d):(?P<seconds_only>[0-5]\d)"
)

EXPERT_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"Expert\s+(?P<number>\d+)"
    r"\s*[-–—:]\s*"
    r"(?P<name>[^,\n]+)"
    r"(?:,\s*(?P<role>[^,\n]+))?"
    r"(?:,\s*(?P<market>[^\n]+))?",
    re.IGNORECASE,
)

SPEAKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<speaker>[^:]{1,100}):\s*(?P<text>.+)$"
)

EXPERT_SPEAKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:dr\.?\s+)?(?P<name>[A-Za-zÀ-ÖØ-öø-ÿ.'’\- ]+)$",
    re.IGNORECASE,
)

INTERVIEWER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(interviewer|interviewer\s+\d+|host|moderator)$",
    re.IGNORECASE,
)


class TranscriptParser:
    """
    Parse raw expert interview transcripts into structured segments.

    The parser keeps the original source file attached to every segment
    and derives timestamps only from timestamps explicitly present in the
    source transcript.
    """

    def parse(
        self,
        text: str,
        source_file: str | Path,
        document_id: str | None = None,
    ) -> tuple[TranscriptMetadata, list[TranscriptSegment]]:
        """
        Parse a complete transcript.

        Args:
            text: Raw transcript text.
            source_file: Original transcript filename or path.
            document_id: Optional stable document identifier.

        Returns:
            A tuple containing transcript metadata and parsed segments.

        Raises:
            SensorException: If parsing fails or no timestamped content
                can be extracted.
        """
        try:
            if not text or not text.strip():
                raise ValueError("Transcript content cannot be empty.")

            source_path = Path(source_file)
            source_name = source_path.name

            metadata = self._extract_metadata(
                text=text,
                source_file=source_name,
                document_id=document_id,
            )

            segments = self._parse_segments(
                text=text,
                metadata=metadata,
            )

            if not segments:
                raise ValueError(
                    f"No timestamped transcript segments found in "
                    f"'{source_name}'."
                )

            metadata.segment_count = len(segments)
            metadata.duration_seconds = self._calculate_duration(segments)

            logger.info(
                "Parsed transcript '%s': %d segments, %.2f seconds",
                source_name,
                len(segments),
                metadata.duration_seconds or 0.0,
            )

            return metadata, segments

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to parse transcript: %s",
                source_file,
            )
            raise SensorException(str(error), _sys_module()) from error

    def _extract_metadata(
        self,
        text: str,
        source_file: str,
        document_id: str | None,
    ) -> TranscriptMetadata:
        """
        Extract expert and market metadata from the transcript header.
        """
        match = EXPERT_HEADER_PATTERN.search(text)

        if match:
            expert_number = int(match.group("number"))
            expert_name = self._clean_text(match.group("name"))
            expert_role = self._clean_text(match.group("role"))
            market = self._clean_text(match.group("market"))

            return TranscriptMetadata(
                document_id=document_id or self._make_document_id(source_file),
                source_file=source_file,
                expert_number=expert_number,
                expert_name=expert_name,
                expert_role=expert_role,
                market=market,
            )

        logger.warning(
            "No expert header found in '%s'. Metadata will remain minimal.",
            source_file,
        )

        return TranscriptMetadata(
            document_id=document_id or self._make_document_id(source_file),
            source_file=source_file,
        )

    def _parse_segments(
        self,
        text: str,
        metadata: TranscriptMetadata,
    ) -> list[TranscriptSegment]:
        """
        Parse timestamped blocks from transcript text.
        """
        lines = text.splitlines()
        timestamped_lines: list[tuple[int, int, str]] = []

        for line_number, line in enumerate(lines):
            match = TIMESTAMP_PATTERN.search(line)

            if not match:
                continue

            timestamp_text = match.group(0)
            timestamp_position = match.start()

            timestamp_seconds = self.parse_timestamp(timestamp_text)

            content_start = match.end()
            content = line[content_start:].strip()

            if not content:
                continue

            timestamped_lines.append(
                (
                    timestamp_seconds,
                    timestamp_position,
                    content,
                )
            )

        segments: list[TranscriptSegment] = []

        for index, (start_seconds, _, content) in enumerate(timestamped_lines):
            next_start = (
                timestamped_lines[index + 1][0]
                if index + 1 < len(timestamped_lines)
                else None
            )

            speaker, spoken_text = self._extract_speaker_and_text(content)

            speaker_type = self._classify_speaker(
                speaker=speaker,
                expert_name=metadata.expert_name,
            )

            end_seconds = self._infer_end_seconds(
                start_seconds=start_seconds,
                next_start=next_start,
            )

            segment = TranscriptSegment(
                segment_id=f"{metadata.document_id}-segment-{index + 1:04d}",
                document_id=metadata.document_id,
                source_file=metadata.source_file,
                expert_name=metadata.expert_name or "Unknown Expert",
                expert_role=metadata.expert_role,
                market=metadata.market,
                speaker=speaker,
                speaker_type=speaker_type,
                text=self._clean_text(spoken_text),
                start_timestamp=self.format_timestamp(start_seconds),
                end_timestamp=self.format_timestamp(end_seconds)
                if end_seconds is not None
                else None,
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds)
                if end_seconds is not None
                else None,
            )

            segments.append(segment)

        return segments

    def _extract_speaker_and_text(
        self,
        content: str,
    ) -> tuple[str, str]:
        """
        Extract speaker name when the line uses 'Speaker: text' syntax.

        If no explicit speaker is present, the entire line is retained
        as transcript text and the speaker is marked unknown.
        """
        match = SPEAKER_PATTERN.match(content.strip())

        if not match:
            return "Unknown", content.strip()

        speaker = self._clean_text(match.group("speaker"))
        spoken_text = match.group("text").strip()

        return speaker, spoken_text

    def _classify_speaker(
        self,
        speaker: str,
        expert_name: str | None,
    ) -> SpeakerType:
        """
        Classify a speaker without guessing beyond available metadata.
        """
        if not speaker or speaker.lower() == "unknown":
            return SpeakerType.UNKNOWN

        if INTERVIEWER_PATTERN.match(speaker.strip()):
            return SpeakerType.INTERVIEWER

        if expert_name:
            normalized_speaker = self._normalize_name(speaker)
            normalized_expert = self._normalize_name(expert_name)

            if (
                normalized_speaker == normalized_expert
                or normalized_speaker in normalized_expert
                or normalized_expert in normalized_speaker
            ):
                return SpeakerType.EXPERT

            expert_last_name = normalized_expert.split()[-1]

            if expert_last_name and expert_last_name in normalized_speaker:
                return SpeakerType.EXPERT

        return SpeakerType.UNKNOWN

    def parse_timestamp(self, timestamp: str) -> int:
        """
        Convert a timestamp string into elapsed seconds.

        Supported formats:
            HH:MM:SS
            MM:SS

        Args:
            timestamp: Timestamp such as '00:18' or '01:02:14'.

        Returns:
            Timestamp expressed as seconds from the beginning.

        Raises:
            ValueError: If the timestamp is malformed.
        """
        normalized = timestamp.strip()

        match = TIMESTAMP_PATTERN.fullmatch(normalized)

        if not match:
            raise ValueError(f"Invalid timestamp format: '{timestamp}'")

        if match.group("hours") is not None:
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))

            return hours * 3600 + minutes * 60 + seconds

        minutes = int(match.group("minutes_only"))
        seconds = int(match.group("seconds_only"))

        return minutes * 60 + seconds

    def format_timestamp(self, seconds: int | float) -> str:
        """
        Convert elapsed seconds to MM:SS or HH:MM:SS format.

        Args:
            seconds: Number of elapsed seconds.

        Returns:
            Human-readable timestamp.
        """
        total_seconds = max(0, int(round(seconds)))

        hours, remainder = divmod(total_seconds, 3600)
        minutes, remaining_seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

        return f"{minutes:02d}:{remaining_seconds:02d}"

    def _infer_end_seconds(
        self,
        start_seconds: int,
        next_start: int | None,
    ) -> int | None:
        """
        Infer a segment end only from the next explicit timestamp.

        No arbitrary timestamp is invented for the final segment.
        """
        if next_start is None:
            return None

        if next_start <= start_seconds:
            logger.warning(
                "Non-increasing transcript timestamp detected: "
                "%s -> %s",
                start_seconds,
                next_start,
            )
            return None

        return next_start

    def _calculate_duration(
        self,
        segments: list[TranscriptSegment],
    ) -> float | None:
        """
        Calculate transcript duration from explicit timestamps.
        """
        if not segments:
            return None

        last_segment = segments[-1]

        if last_segment.end_seconds is not None:
            return last_segment.end_seconds

        return last_segment.start_seconds

    def _make_document_id(self, source_file: str) -> str:
        """
        Generate a deterministic document identifier from the filename.
        """
        stem = Path(source_file).stem
        normalized = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-")

        return normalized.lower() or "transcript"

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        """
        Normalize whitespace without changing lexical content.
        """
        if value is None:
            return None

        cleaned = " ".join(value.split())
        return cleaned if cleaned else None

    @staticmethod
    def _normalize_name(value: str) -> str:
        """
        Normalize a speaker/expert name for comparison.
        """
        value = value.lower().replace("’", "'")
        value = re.sub(r"\bdr\.?\b", "", value)
        value = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s'-]", "", value)
        return " ".join(value.split())


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "TIMESTAMP_PATTERN",
    "TranscriptParser",
]