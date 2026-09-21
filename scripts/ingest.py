"""
Transcript ingestion and vector-indexing pipeline.

Usage:
    python scripts/ingest.py

Optional:
    python scripts/ingest.py --clear
    python scripts/ingest.py --data-dir data/raw
    python scripts/ingest.py --force

Pipeline:

    Raw transcripts
        ↓
    TranscriptLoader
        ↓
    TranscriptParser
        ↓
    Timestamp-aware Chunker
        ↓
    EmbeddingService
        ↓
    FAISSVectorStore
        ↓
    Persistent vector index

The script intentionally contains no transcript-specific hard-coded
content. Any .txt or .md transcript placed in the configured raw-data
directory can be processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from logger import logging

from src.config import Settings, get_settings
from src.ingestion.chunker import TranscriptChunker
from src.ingestion.loader import TranscriptLoader
from src.ingestion.parser import TranscriptParser
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore


logger = logging.getLogger(__name__)


class IngestionPipelineError(RuntimeError):
    """Raised when transcript ingestion cannot be completed."""


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute the transcript ingestion pipeline.

    Args:
        argv:
            Optional command-line arguments. When omitted, arguments are
            read from sys.argv.

    Returns:
        Process exit code.
    """
    args = _parse_arguments(argv)

    try:
        settings = get_settings()
        settings.ensure_directories()

        return _run_pipeline(
            settings=settings,
            data_directory=args.data_dir,
            clear_existing=args.clear,
            force=args.force,
        )

    except KeyboardInterrupt:
        logger.warning(
            "Ingestion interrupted by user."
        )

        print(
            "\nIngestion interrupted."
        )

        return 130

    except Exception as error:
        logger.exception(
            "Transcript ingestion failed."
        )

        print(
            "\nERROR: Transcript ingestion failed."
        )

        print(
            f"Details: {error}"
        )

        return 1


def _run_pipeline(
    *,
    settings: Settings,
    data_directory: str | None,
    clear_existing: bool,
    force: bool,
) -> int:
    """
    Run all ingestion stages.
    """
    raw_directory = _resolve_data_directory(
        settings,
        data_directory,
    )

    logger.info(
        "Starting transcript ingestion from: %s",
        raw_directory,
    )

    print(
        "=" * 72
    )

    print(
        "Hasamex Expert Analysis - Transcript Ingestion"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSource directory: {raw_directory}"
    )

    # ------------------------------------------------------------------
    # Initialize ingestion components
    # ------------------------------------------------------------------

    loader = TranscriptLoader()

    parser = TranscriptParser()

    chunker = TranscriptChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # EmbeddingService expects Settings as its constructor argument.
    embedding_service = EmbeddingService(
        settings
    )

    # FAISSVectorStore also expects the embedding service and settings.
    vector_store = FAISSVectorStore(
        embedding_service=embedding_service,
        settings=settings,
    )

    # ------------------------------------------------------------------
    # Optional clearing of existing vector store
    # ------------------------------------------------------------------

    if clear_existing:
        _clear_vector_store(
            vector_store
        )

    # ------------------------------------------------------------------
    # Load transcript files
    # ------------------------------------------------------------------

    files = loader.load_directory(
        raw_directory
    )

    if not files:
        raise IngestionPipelineError(
            f"No supported transcript files were found in "
            f"'{raw_directory}'. "
            f"Expected .txt or .md files."
        )

    print(
        f"Found {len(files)} transcript file(s)."
    )

    # ------------------------------------------------------------------
    # Process every transcript
    #
    # TranscriptLoader.load_directory() returns:
    #
    #     {
    #         "file1.txt": "transcript content...",
    #         "file2.txt": "transcript content...",
    #     }
    #
    # Therefore we must iterate over files.items()
    # rather than directly over files.
    # ------------------------------------------------------------------

    all_chunks = []

    for file_index, (
        filename,
        transcript_content,
    ) in enumerate(
        files.items(),
        start=1,
    ):
        print(
            f"\n[{file_index}/{len(files)}] "
            f"Processing: {filename}"
        )

        try:
            # ----------------------------------------------------------
            # Parse transcript content
            # ----------------------------------------------------------

            metadata, segments = parser.parse(
                text=transcript_content,
                source_file=filename,
            )

            if not segments:
                logger.warning(
                    "No transcript segments found in %s.",
                    filename,
                )

                print(
                    "  Warning: no transcript segments found."
                )

                continue

            # ----------------------------------------------------------
            # Create timestamp-aware chunks
            # ----------------------------------------------------------

            chunks = chunker.chunk(
                metadata=metadata,
                segments=segments,
            )

            if not chunks:
                logger.warning(
                    "No chunks generated for %s.",
                    filename,
                )

                print(
                    "  Warning: no chunks generated."
                )

                continue

            # ----------------------------------------------------------
            # Add chunks to global collection
            # ----------------------------------------------------------

            all_chunks.extend(
                chunks
            )

            # ----------------------------------------------------------
            # Display transcript statistics
            # ----------------------------------------------------------

            print(
                f"  Expert: "
                f"{metadata.expert_name or 'Unknown'}"
            )

            print(
                f"  Market: "
                f"{metadata.market or 'Unknown'}"
            )

            print(
                f"  Segments: {len(segments)}"
            )

            print(
                f"  Chunks: {len(chunks)}"
            )

            logger.info(
                "Processed %s: %d segments -> %d chunks.",
                filename,
                len(segments),
                len(chunks),
            )

        except Exception as error:
            logger.exception(
                "Failed to process transcript: %s",
                filename,
            )

            raise IngestionPipelineError(
                f"Failed to process transcript "
                f"'{filename}'."
            ) from error

    # ------------------------------------------------------------------
    # Validate generated chunks
    # ------------------------------------------------------------------

    if not all_chunks:
        raise IngestionPipelineError(
            "The ingestion pipeline did not produce any "
            "timestamp-aware transcript chunks."
        )

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # ------------------------------------------------------------------
    # Generate embeddings and build FAISS index
    # ------------------------------------------------------------------

    _build_vector_store(
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunks=all_chunks,
        force=force,
    )

    # ------------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------------

    _print_summary(
        vector_store=vector_store,
        chunk_count=len(all_chunks),
    )

    logger.info(
        "Transcript ingestion completed successfully."
    )

    return 0


def _build_vector_store(
    *,
    vector_store: FAISSVectorStore,
    embedding_service: EmbeddingService,
    chunks: list,
    force: bool,
) -> None:
    """
    Generate embeddings and build/persist the vector index.
    """

    print(
        "\nGenerating embeddings..."
    )

    # Extract chunk text for embedding generation.
    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedding_service.embed_documents(
        texts
    )

    # Validate embedding count.
    if len(embeddings) != len(chunks):
        raise IngestionPipelineError(
            "The number of generated embeddings does not "
            "match the number of transcript chunks."
        )

    print(
        f"Generated {len(embeddings)} embedding(s)."
    )

    # ------------------------------------------------------------------
    # Clear existing vector store when force rebuild is requested.
    # ------------------------------------------------------------------

    if force and vector_store.exists():
        logger.info(
            "Force rebuild requested. "
            "Clearing existing vector store."
        )

        vector_store.clear()

    # ------------------------------------------------------------------
    # Replace existing vector store.
    # ------------------------------------------------------------------

    if vector_store.exists():
        logger.info(
            "Existing vector store found. "
            "Replacing index."
        )

        vector_store.clear()

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------

    print(
        "Building FAISS vector index..."
    )

    vector_store.build(
        chunks=chunks,
        embeddings=embeddings,
    )

    # ------------------------------------------------------------------
    # Persist index and metadata
    # ------------------------------------------------------------------

    print(
        "Persisting vector index..."
    )

    vector_store.save()

    logger.info(
        "Vector store successfully built and saved."
    )


def _clear_vector_store(
    vector_store: FAISSVectorStore,
) -> None:
    """
    Remove an existing vector store.
    """

    if not vector_store.exists():
        print(
            "\nNo existing vector store to clear."
        )

        return

    print(
        "\nClearing existing vector store..."
    )

    vector_store.clear()

    logger.info(
        "Existing vector store cleared."
    )


def _resolve_data_directory(
    settings: Settings,
    data_directory: str | None,
) -> Path:
    """
    Resolve and validate the transcript input directory.
    """

    if data_directory:
        path = (
            Path(data_directory)
            .expanduser()
            .resolve()
        )
    else:
        path = settings.raw_data_path

    if not path.exists():
        raise IngestionPipelineError(
            f"Transcript directory does not exist: {path}"
        )

    if not path.is_dir():
        raise IngestionPipelineError(
            f"Transcript input path is not a directory: {path}"
        )

    return path


def _print_summary(
    *,
    vector_store: FAISSVectorStore,
    chunk_count: int,
) -> None:
    """
    Print a concise ingestion summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Indexed chunks : {chunk_count}"
    )

    # FAISSVectorStore.size is a property, not a method.
    try:
        vector_count = vector_store.size
    except Exception:
        vector_count = chunk_count

    print(
        f"Vector entries : {vector_count}"
    )

    # FAISSVectorStore uses storage_path.
    print(
        f"Vector store   : {vector_store.storage_path}"
    )

    print(
        "\nThe application can now be started with:"
    )

    print(
        "  streamlit run app.py"
    )

    print(
        "=" * 72
    )


def _parse_arguments(
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Parse Hasamex expert transcripts and build "
            "the persistent FAISS vector index."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory containing transcript files. "
            "Defaults to RAW_DATA_DIR from the application settings."
        ),
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Clear the existing vector store before ingestion."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force rebuilding the vector index even when an "
            "existing index is present."
        ),
    )

    return parser.parse_args(
        argv
    )


if __name__ == "__main__":
    sys.exit(
        main()
    )