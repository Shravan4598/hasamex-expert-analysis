"""
Application configuration for the Hasamex Expert Analysis project.

All configurable values are loaded from environment variables and/or
a local .env file. Secrets such as API keys must never be hard-coded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root:
# src/config.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------
    app_name: str = Field(
        default="Hasamex Expert Analysis",
        validation_alias="APP_NAME",
    )

    app_env: str = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------
    google_api_key: str = Field(
        default="",
        validation_alias="GOOGLE_API_KEY",
    )

    llm_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="LLM_MODEL",
    )

    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        validation_alias="LLM_TEMPERATURE",
    )

    llm_max_output_tokens: int = Field(
        default=2048,
        gt=0,
        validation_alias="LLM_MAX_OUTPUT_TOKENS",
    )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    retrieval_top_k: int = Field(
        default=8,
        gt=0,
        validation_alias="RETRIEVAL_TOP_K",
    )

    rerank_top_k: int = Field(
        default=5,
        gt=0,
        validation_alias="RERANK_TOP_K",
    )

    min_retrieval_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        validation_alias="MIN_RETRIEVAL_SCORE",
    )

    min_evidence_coverage: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        validation_alias="MIN_EVIDENCE_COVERAGE",
    )

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------
    chunk_size: int = Field(
        default=1200,
        gt=0,
        validation_alias="CHUNK_SIZE",
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        validation_alias="CHUNK_OVERLAP",
    )

    # ---------------------------------------------------------
    # Data / storage paths
    # ---------------------------------------------------------
    raw_data_dir: str = Field(
        default="data/raw",
        validation_alias="RAW_DATA_DIR",
    )

    processed_data_dir: str = Field(
        default="data/processed",
        validation_alias="PROCESSED_DATA_DIR",
    )

    vector_store_dir: str = Field(
        default="storage/vector_store",
        validation_alias="VECTOR_STORE_DIR",
    )

    # ---------------------------------------------------------
    # Streamlit
    # ---------------------------------------------------------
    streamlit_page_title: str = Field(
        default="Hasamex Expert Analysis",
        validation_alias="STREAMLIT_PAGE_TITLE",
    )

    streamlit_page_icon: str = Field(
        default="🔎",
        validation_alias="STREAMLIT_PAGE_ICON",
    )

    # ---------------------------------------------------------
    # Validators
    # ---------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(
        cls,
        value: int,
        info,
    ) -> int:
        """Ensure chunk overlap is smaller than chunk size."""

        chunk_size = info.data.get("chunk_size")

        if chunk_size is not None and value >= chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        return value

    @field_validator("rerank_top_k")
    @classmethod
    def validate_rerank_top_k(
        cls,
        value: int,
        info,
    ) -> int:
        """Ensure reranking does not request more items than retrieval."""

        retrieval_top_k = info.data.get("retrieval_top_k")

        if (
            retrieval_top_k is not None
            and value > retrieval_top_k
        ):
            raise ValueError(
                "RERANK_TOP_K cannot be greater than RETRIEVAL_TOP_K."
            )

        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate and normalize the configured log level."""

        normalized = value.upper()

        allowed_levels = {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }

        if normalized not in allowed_levels:
            raise ValueError(
                "LOG_LEVEL must be one of: "
                "CRITICAL, ERROR, WARNING, INFO, DEBUG."
            )

        return normalized

    # ---------------------------------------------------------
    # Resolved paths
    # ---------------------------------------------------------
    @property
    def raw_data_path(self) -> Path:
        """Return the absolute path to raw transcript data."""

        return self._resolve_project_path(self.raw_data_dir)

    @property
    def processed_data_path(self) -> Path:
        """Return the absolute path to processed data."""

        return self._resolve_project_path(self.processed_data_dir)

    @property
    def vector_store_path(self) -> Path:
        """Return the absolute path to the vector store."""

        return self._resolve_project_path(self.vector_store_dir)

    def _resolve_project_path(self, configured_path: str) -> Path:
        """
        Resolve a configured path relative to the project root.

        Absolute paths are preserved. Relative paths are interpreted
        relative to the project root.
        """

        path = Path(configured_path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def ensure_directories(self) -> None:
        """Create required application directories if they do not exist."""

        self.raw_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store_path.mkdir(
            parents=True,
            exist_ok=True,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings.

    Caching ensures the same validated configuration object is reused
    throughout the application lifecycle.
    """

    settings = Settings()
    settings.ensure_directories()

    return settings