"""
Transcript file loading utilities.

This module is responsible only for reading transcript files from disk.
Parsing transcript structure and creating timestamp-aware segments are
handled by downstream ingestion modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from exception import SensorException
from logger import logging


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".txt", ".md"})


class TranscriptLoader:
    """Load transcript files from the local filesystem."""

    def __init__(self, supported_extensions: frozenset[str] | None = None) -> None:
        """
        Initialize the transcript loader.

        Args:
            supported_extensions: Optional set of file extensions that the
                loader accepts. Defaults to .txt and .md.
        """
        self.supported_extensions = (
            supported_extensions
            if supported_extensions is not None
            else SUPPORTED_EXTENSIONS
        )

    def load_file(self, file_path: str | Path) -> str:
        """
        Read a single transcript file as UTF-8 text.

        Args:
            file_path: Path to the transcript file.

        Returns:
            The complete transcript content.

        Raises:
            SensorException: If the path is invalid, unsupported, unreadable,
                or contains no usable text.
        """
        try:
            path = Path(file_path)

            self._validate_path(path)

            content = path.read_text(encoding="utf-8")

            if not content.strip():
                raise ValueError(f"Transcript file is empty: {path}")

            logger.info(
                "Successfully loaded transcript: %s (%d characters)",
                path,
                len(content),
            )

            return content

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Failed to load transcript file: %s", file_path)
            raise SensorException(str(error), sys_module()) from error

    def load_directory(self, directory_path: str | Path) -> dict[str, str]:
        """
        Load all supported transcript files from a directory.

        Files are returned in deterministic alphabetical order.

        Args:
            directory_path: Directory containing transcript files.

        Returns:
            Mapping of filename to transcript content.

        Raises:
            SensorException: If the directory does not exist or cannot be read.
        """
        try:
            directory = Path(directory_path)

            if not directory.exists():
                raise FileNotFoundError(
                    f"Transcript directory does not exist: {directory}"
                )

            if not directory.is_dir():
                raise NotADirectoryError(
                    f"Transcript path is not a directory: {directory}"
                )

            files = sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in self.supported_extensions
                ),
                key=lambda path: path.name.lower(),
            )

            transcripts: dict[str, str] = {}

            for path in files:
                transcripts[path.name] = self.load_file(path)

            logger.info(
                "Loaded %d transcript file(s) from directory: %s",
                len(transcripts),
                directory,
            )

            return transcripts

        except SensorException:
            raise
        except Exception as error:
            logger.exception(
                "Failed to load transcript directory: %s",
                directory_path,
            )
            raise SensorException(str(error), sys_module()) from error

    def validate_file(self, file_path: str | Path) -> bool:
        """
        Validate whether a path is a supported transcript file.

        Args:
            file_path: Path to validate.

        Returns:
            True when the path exists, is a regular file, and has a
            supported extension.
        """
        try:
            path = Path(file_path)

            return (
                path.exists()
                and path.is_file()
                and path.suffix.lower() in self.supported_extensions
            )
        except (OSError, TypeError, ValueError):
            return False

    def _validate_path(self, path: Path) -> None:
        """
        Validate a transcript path before reading it.

        Args:
            path: Path to validate.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path points to a directory.
            ValueError: If the extension is unsupported.
        """
        if not path.exists():
            raise FileNotFoundError(f"Transcript file does not exist: {path}")

        if not path.is_file():
            raise IsADirectoryError(f"Transcript path is not a file: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            supported = ", ".join(sorted(self.supported_extensions))
            raise ValueError(
                f"Unsupported transcript format '{path.suffix}'. "
                f"Supported formats: {supported}"
            )


def sys_module():
    """
    Return the sys module for compatibility with the project's exception API.

    The provided SensorException expects the active sys module so that its
    traceback information can be extracted by exception.py.
    """
    import sys

    return sys


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "TranscriptLoader",
]