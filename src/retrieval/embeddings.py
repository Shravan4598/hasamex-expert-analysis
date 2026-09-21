"""
Embedding generation for transcript retrieval.

This module wraps Sentence Transformers behind a small application-level
interface so the rest of the retrieval pipeline does not depend directly
on the embedding library.

Embeddings are normalized because the vector store uses inner-product
similarity, which is equivalent to cosine similarity for normalized vectors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from exception import SensorException
from logger import logging

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generate dense vector embeddings for transcript chunks and queries.

    The model is loaded lazily so importing the application does not
    immediately download or initialize a potentially large model.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the embedding service.

        Args:
            settings: Optional application settings. When omitted, the
                project's cached settings are used.
        """
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazily load and cache the Sentence Transformer model.

        Returns:
            Initialized SentenceTransformer instance.

        Raises:
            SensorException: If the model cannot be initialized.
        """
        if self._model is None:
            try:
                logger.info(
                    "Loading embedding model: %s",
                    self.model_name,
                )

                self._model = SentenceTransformer(self.model_name)

                logger.info(
                    "Embedding model loaded successfully: %s",
                    self.model_name,
                )

            except Exception as error:
                logger.exception(
                    "Failed to load embedding model: %s",
                    self.model_name,
                )
                raise SensorException(
                    str(error),
                    _sys_module(),
                ) from error

        return self._model

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding model.

        Returns:
            Number of dimensions in each embedding vector.
        """
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception as error:
            logger.exception("Unable to determine embedding dimension.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding for one text string.

        Args:
            text: Text to embed.

        Returns:
            One-dimensional float32 NumPy array.

        Raises:
            SensorException: If the input or embedding operation is invalid.
        """
        try:
            self._validate_text(text)

            embeddings = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vector = np.asarray(
                embeddings[0],
                dtype=np.float32,
            )

            return vector

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Failed to generate text embedding.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate normalized embeddings for multiple texts.

        Args:
            texts: Sequence of text strings.
            batch_size: Number of texts processed per embedding batch.

        Returns:
            Two-dimensional float32 NumPy array with shape:
            (number_of_texts, embedding_dimension).

        Raises:
            SensorException: If input validation or embedding fails.
        """
        try:
            if not texts:
                return np.empty(
                    (0, self.dimension),
                    dtype=np.float32,
                )

            if batch_size <= 0:
                raise ValueError("batch_size must be greater than zero.")

            normalized_texts = [
                self._validate_and_normalize_text(text)
                for text in texts
            ]

            logger.info(
                "Generating embeddings for %d text(s), batch_size=%d",
                len(normalized_texts),
                batch_size,
            )

            embeddings = self.model.encode(
                normalized_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            result = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            if result.ndim != 2:
                raise ValueError(
                    "Embedding model returned an unexpected array shape: "
                    f"{result.shape}"
                )

            logger.info(
                "Generated embedding matrix with shape %s",
                result.shape,
            )

            return result

        except SensorException:
            raise
        except Exception as error:
            logger.exception("Failed to generate text embeddings.")
            raise SensorException(
                str(error),
                _sys_module(),
            ) from error

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding specifically for a retrieval query.

        Args:
            query: User search question.

        Returns:
            Normalized query embedding.
        """
        return self.embed_text(query)

    def embed_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for documents/chunks.

        Args:
            documents: Text documents or transcript chunks.
            batch_size: Number of documents processed per batch.

        Returns:
            Normalized document embedding matrix.
        """
        return self.embed_texts(
            texts=documents,
            batch_size=batch_size,
        )

    def get_model_info(self) -> dict[str, str | int]:
        """
        Return metadata useful for diagnostics and index validation.

        Returns:
            Dictionary containing model name and vector dimension.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    @staticmethod
    def _validate_text(text: str) -> None:
        """
        Validate a single text input.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Embedding input must be a string, got {type(text).__name__}."
            )

        if not text.strip():
            raise ValueError("Cannot generate an embedding for empty text.")

    def _validate_and_normalize_text(self, text: str) -> str:
        """
        Validate and normalize text without changing its semantic content.
        """
        self._validate_text(text)
        return text.strip()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Return the application's cached embedding service.

    Returns:
        Singleton-like EmbeddingService instance for the current process.
    """
    return EmbeddingService(get_settings())


def _sys_module():
    """Return the active sys module for SensorException."""
    import sys

    return sys


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
]