"""
Transcript parsing and timestamp extraction.

The parser converts raw expert-call transcripts into structured,
timestamp-aware TranscriptSegment objects.

Expected transcript format:

    Expert 1 – Dr. Jean Martin
    Role: Head of Urology
    Market: France

    00:00
    Interviewer: Thanks for joining...

    00:18
    Dr. Martin: Adoption is growing...

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
    r"^\s*Expert\s+(?P<number>\d+)"
    r"\s*[-–—:]\s*"
    r"(?P<name>[^\r\n]+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


ROLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*Role\s*:\s*(?P<role>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


MARKET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*Market\s*:\s*(?P<market>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


SPEAKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<speaker>[^:]{1,100}):\s*(?P<text>.+)$"
)


INTERVIEWER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(interviewer|interviewer\s+\d+|host|moderator)$",
    re.IGNORECASE,
)


class TranscriptParser:
    """
    Parse raw expert interview transcripts into structured segments.

    The parser keeps the original source file attached to every segment
    and derives timestamps only from timestamps explicitly present in
    the source transcript.
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
            text:
                Raw transcript text.

            source_file:
                Original transcript filename or path.

            document_id:
                Optional stable document identifier.

        Returns:
            A tuple containing transcript metadata and parsed segments.

        Raises:
            SensorException:
                If parsing fails or no timestamped content can be
                extracted.
        """
        try:
            if not text or not text.strip():
                raise ValueError(
                    "Transcript content cannot be empty."
                )

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
                    f"No timestamped transcript segments found "
                    f"in '{source_name}'."
                )

            logger.info(
                "Parsed transcript '%s': %d segments",
                source_name,
                len(segments),
            )

            return metadata, segments

        except SensorException:
            raise

        except Exception as error:
            logger.exception(
                "Failed to parse transcript: %s",
                source_file,
            )

            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def _extract_metadata(
        self,
        text: str,
        source_file: str,
        document_id: str | None,
    ) -> TranscriptMetadata:
        """
        Extract expert metadata from the transcript header.

        Expected format:

            Expert 1 – Dr. Jean Martin
            Role: Head of Urology
            Market: France

        The expert number is intentionally not stored because the
        TranscriptMetadata domain model does not define an expert_number
        field.
        """

        expert_match = EXPERT_HEADER_PATTERN.search(text)

        if not expert_match:
            raise ValueError(
                f"Could not find expert header in '{source_file}'."
            )

        expert_name = self._clean_required_text(
            expert_match.group("name")
        )

        role_match = ROLE_PATTERN.search(text)
        market_match = MARKET_PATTERN.search(text)

        if not role_match:
            raise ValueError(
                f"Could not find 'Role:' metadata in "
                f"'{source_file}'."
            )

        if not market_match:
            raise ValueError(
                f"Could not find 'Market:' metadata in "
                f"'{source_file}'."
            )

        expert_role = self._clean_required_text(
            role_match.group("role")
        )

        market = self._clean_required_text(
            market_match.group("market")
        )

        return TranscriptMetadata(
            document_id=(
                document_id
                or self._make_document_id(source_file)
            ),
            source_file=source_file,
            expert_name=expert_name,
            expert_role=expert_role,
            market=market,
        )

    def _parse_segments(
        self,
        text: str,
        metadata: TranscriptMetadata,
    ) -> list[TranscriptSegment]:
        """
        Parse timestamped transcript blocks.

        The actual transcript format has timestamps on their own lines:

            00:00
            Interviewer: Question...

            00:18
            Dr. Martin: Answer...
        """

        lines = text.splitlines()

        timestamped_lines: list[
            tuple[int, int]
        ] = []

        for line_number, line in enumerate(lines):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            match = TIMESTAMP_PATTERN.fullmatch(
                stripped_line
            )

            if not match:
                continue

            timestamp_seconds = self.parse_timestamp(
                stripped_line
            )

            timestamped_lines.append(
                (
                    timestamp_seconds,
                    line_number,
                )
            )

        segments: list[TranscriptSegment] = []

        for index, (
            start_seconds,
            line_number,
        ) in enumerate(timestamped_lines):

            next_start = (
                timestamped_lines[index + 1][0]
                if index + 1 < len(timestamped_lines)
                else None
            )

            content = self._collect_segment_content(
                lines=lines,
                start_line=line_number,
                end_line=(
                    timestamped_lines[index + 1][1]
                    if index + 1 < len(timestamped_lines)
                    else len(lines)
                ),
            )

            if not content:
                logger.warning(
                    "Skipping timestamp %s in %s because "
                    "no transcript content was found.",
                    self.format_timestamp(start_seconds),
                    metadata.source_file,
                )
                continue

            speaker, spoken_text = (
                self._extract_speaker_and_text(content)
            )

            if not spoken_text:
                logger.warning(
                    "Skipping empty transcript segment at %s "
                    "in %s.",
                    self.format_timestamp(start_seconds),
                    metadata.source_file,
                )
                continue

            speaker_type = self._classify_speaker(
                speaker=speaker,
                expert_name=metadata.expert_name,
            )

            end_seconds = self._infer_end_seconds(
                start_seconds=start_seconds,
                next_start=next_start,
            )

            segment = TranscriptSegment(
                segment_id=(
                    f"{metadata.document_id}"
                    f"-segment-{index + 1:04d}"
                ),
                document_id=metadata.document_id,
                source_file=metadata.source_file,
                expert_name=metadata.expert_name,
                expert_role=metadata.expert_role,
                market=metadata.market,
                speaker=speaker,
                speaker_type=speaker_type,
                text=self._clean_required_text(
                    spoken_text
                ),
                start_timestamp=self.format_timestamp(
                    start_seconds
                ),
                end_timestamp=(
                    self.format_timestamp(end_seconds)
                    if end_seconds is not None
                    else None
                ),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )

            segments.append(segment)

        return segments

    def _collect_segment_content(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
    ) -> str:
        """
        Collect all non-empty lines between two timestamps.

        Example:

            00:18
            Dr. Martin: Adoption is growing...

        becomes:

            Dr. Martin: Adoption is growing...
        """

        content_lines: list[str] = []

        for line in lines[start_line + 1:end_line]:
            stripped = line.strip()

            if not stripped:
                continue

            content_lines.append(stripped)

        return " ".join(content_lines)

    def _extract_speaker_and_text(
        self,
        content: str,
    ) -> tuple[str, str]:
        """
        Extract speaker name from:

            Speaker: transcript text

        If no speaker label exists, the speaker is marked unknown.
        """

        match = SPEAKER_PATTERN.match(
            content.strip()
        )

        if not match:
            return (
                "Unknown",
                content.strip(),
            )

        speaker = self._clean_required_text(
            match.group("speaker")
        )

        spoken_text = self._clean_required_text(
            match.group("text")
        )

        return speaker, spoken_text

    def _classify_speaker(
        self,
        speaker: str,
        expert_name: str,
    ) -> SpeakerType:
        """
        Classify a speaker based on explicit transcript labels.

        Interviewer labels are recognized directly.

        Expert labels are matched against the expert's full name,
        including common abbreviated forms such as:

            Dr. Jean Martin
            Dr. Martin
            Jean Martin
            Martin
        """

        if not speaker:
            return SpeakerType.UNKNOWN

        normalized_speaker = self._normalize_name(
            speaker
        )

        if not normalized_speaker:
            return SpeakerType.UNKNOWN

        if INTERVIEWER_PATTERN.match(
            speaker.strip()
        ):
            return SpeakerType.INTERVIEWER

        normalized_expert = self._normalize_name(
            expert_name
        )

        if not normalized_expert:
            return SpeakerType.UNKNOWN

        expert_tokens = normalized_expert.split()
        speaker_tokens = normalized_speaker.split()

        if (
            normalized_speaker == normalized_expert
            or normalized_speaker in normalized_expert
            or normalized_expert in normalized_speaker
        ):
            return SpeakerType.EXPERT

        if (
            len(expert_tokens) >= 2
            and expert_tokens[-1] in speaker_tokens
        ):
            return SpeakerType.EXPERT

        return SpeakerType.UNKNOWN

    def parse_timestamp(
        self,
        timestamp: str,
    ) -> int:
        """
        Convert a timestamp into elapsed seconds.

        Supported formats:

            MM:SS
            HH:MM:SS
        """

        normalized = timestamp.strip()

        match = TIMESTAMP_PATTERN.fullmatch(
            normalized
        )

        if not match:
            raise ValueError(
                f"Invalid timestamp format: "
                f"'{timestamp}'"
            )

        if match.group("hours") is not None:
            hours = int(
                match.group("hours")
            )

            minutes = int(
                match.group("minutes")
            )

            seconds = int(
                match.group("seconds")
            )

            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

        minutes = int(
            match.group("minutes_only")
        )

        seconds = int(
            match.group("seconds_only")
        )

        return minutes * 60 + seconds

    def format_timestamp(
        self,
        seconds: int | float,
    ) -> str:
        """
        Convert elapsed seconds to MM:SS or HH:MM:SS.
        """

        total_seconds = max(
            0,
            int(round(seconds)),
        )

        hours, remainder = divmod(
            total_seconds,
            3600,
        )

        minutes, remaining_seconds = divmod(
            remainder,
            60,
        )

        if hours > 0:
            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{remaining_seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    def _infer_end_seconds(
        self,
        start_seconds: int,
        next_start: int | None,
    ) -> int | None:
        """
        Infer a segment end only from the next explicit timestamp.

        The final segment intentionally has no invented end timestamp.
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

    def _make_document_id(
        self,
        source_file: str,
    ) -> str:
        """
        Generate a deterministic document identifier.
        """

        stem = Path(source_file).stem

        normalized = re.sub(
            r"[^a-zA-Z0-9]+",
            "-",
            stem,
        ).strip("-")

        return (
            normalized.lower()
            or "transcript"
        )

    @staticmethod
    def _clean_required_text(
        value: str | None,
    ) -> str:
        """
        Normalize whitespace and require non-empty text.
        """

        if value is None:
            raise ValueError(
                "Required transcript metadata/text is missing."
            )

        cleaned = " ".join(
            value.split()
        )

        if not cleaned:
            raise ValueError(
                "Required transcript metadata/text is empty."
            )

        return cleaned

    @staticmethod
    def _normalize_name(
        value: str,
    ) -> str:
        """
        Normalize names for speaker comparison.
        """

        value = (
            value.lower()
            .replace("’", "'")
        )

        value = re.sub(
            r"\bdr\.?\b",
            "",
            value,
        )

        value = re.sub(
            r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s'-]",
            "",
            value,
        )

        return " ".join(
            value.split()
        )


def _sys_module():
    """Return the active sys module for SensorException."""

    import sys

    return sys


__all__ = [
    "TIMESTAMP_PATTERN",
    "TranscriptParser",
]