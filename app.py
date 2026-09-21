"""
Main Streamlit application for Hasamex Expert Analysis.

This module is the application entry point.

Responsibilities:
    - Configure the Streamlit page.
    - Initialize application services.
    - Load the indexed transcript corpus.
    - Provide top-level navigation.
    - Route users to the appropriate UI workflow.
    - Handle application-level errors safely.

The application does not contain transcript-specific facts or answers.
All analysis is delegated to the ingestion, retrieval, evidence, and
analysis layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from logger import logging

from src.analysis.disagreements import DisagreementAnalyzer
from src.analysis.interview_guide import InterviewGuideAnalyzer
from src.analysis.llm import GeminiLLM
from src.analysis.qa import TranscriptQA
from src.analysis.themes import ThemeAnalyzer
from src.config import Settings, get_settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FAISSVectorStore
from src.ui.dashboard import (
    render_dashboard,
    render_sidebar,
)
from src.ui.disagreements import (
    render_disagreements,
)
from src.ui.interview_guide import (
    render_interview_guide,
)
from src.ui.qa import (
    render_qa,
)
from src.ui.sources import (
    render_sources,
)
from src.ui.themes import (
    render_themes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationServices:
    """
    Container for application-level dependencies.

    Keeping the services together makes dependency wiring explicit and
    prevents individual Streamlit pages from constructing their own
    copies of the retrieval and LLM infrastructure.
    """

    settings: Settings
    embeddings: EmbeddingService
    vector_store: FAISSVectorStore
    retriever: Retriever
    reranker: Reranker
    llm: GeminiLLM
    interview_guide: InterviewGuideAnalyzer
    themes: ThemeAnalyzer
    disagreements: DisagreementAnalyzer
    qa: TranscriptQA


class ApplicationInitializationError(RuntimeError):
    """
    Raised when the application cannot initialize its core services.
    """


def main() -> None:
    """
    Run the Streamlit application.
    """
    settings = _load_settings()

    _configure_page(
        settings
    )

    _render_application_header(
        settings
    )

    try:
        services = _get_application_services(
            settings
        )
    except Exception as error:
        _handle_initialization_error(
            error
        )
        return

    navigation = render_sidebar()

    _render_page(
        navigation=navigation,
        services=services,
    )


@st.cache_resource(
    show_spinner=False,
)
def _get_application_services(
    settings: Settings,
) -> ApplicationServices:
    """
    Build and cache the application's shared services.

    Streamlit reruns the script frequently. Heavy services such as
    embedding models, FAISS indexes, and LLM clients therefore need to
    be cached at the resource level.
    """
    logger.info(
        "Initializing Hasamex Expert Analysis services."
    )

    try:
        settings.ensure_directories()

        embeddings = EmbeddingService(
            model_name=settings.embedding_model,
        )

        vector_store = FAISSVectorStore(
            storage_dir=settings.vector_store_path,
            embedding_model=settings.embedding_model,
        )

        if not vector_store.exists():
            raise ApplicationInitializationError(
                "The vector store has not been built yet. "
                "Run the ingestion/indexing pipeline before starting "
                "the application."
            )

        vector_store.load()

        retriever = Retriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            top_k=settings.retrieval_top_k,
            min_score=settings.min_retrieval_score,
        )

        reranker = Reranker(
            top_k=settings.rerank_top_k,
        )

        llm = GeminiLLM(
            api_key=settings.google_api_key,
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )

        interview_guide = InterviewGuideAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            min_evidence_coverage=(
                settings.min_evidence_coverage
            ),
        )

        themes = ThemeAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            min_evidence_coverage=(
                settings.min_evidence_coverage
            ),
        )

        disagreements = DisagreementAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            min_evidence_coverage=(
                settings.min_evidence_coverage
            ),
        )

        qa = TranscriptQA(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            min_evidence_coverage=(
                settings.min_evidence_coverage
            ),
        )

        logger.info(
            "Application services initialized successfully."
        )

        return ApplicationServices(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            interview_guide=interview_guide,
            themes=themes,
            disagreements=disagreements,
            qa=qa,
        )

    except ApplicationInitializationError:
        raise

    except Exception as error:
        logger.exception(
            "Application service initialization failed."
        )

        raise ApplicationInitializationError(
            "The application services could not be initialized."
        ) from error


def _load_settings() -> Settings:
    """
    Load validated application settings.
    """
    try:
        return get_settings()

    except Exception as error:
        logger.exception(
            "Failed to load application settings."
        )

        st.set_page_config(
            page_title="Hasamex Expert Analysis",
            page_icon="🔎",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.error(
            "Application configuration could not be loaded."
        )

        with st.expander(
            "Configuration details"
        ):
            st.code(
                str(error)
            )

        st.stop()

        raise


def _configure_page(
    settings: Settings,
) -> None:
    """
    Configure the global Streamlit page.
    """
    st.set_page_config(
        page_title=settings.streamlit_page_title,
        page_icon=settings.streamlit_page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_application_header(
    settings: Settings,
) -> None:
    """
    Render the global application header.
    """
    st.markdown(
        """
        <style>
        .hasamex-header {
            padding: 0.25rem 0 0.75rem 0;
        }

        .hasamex-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: -0.5rem;
        }

        .hasamex-traceability {
            border-left: 4px solid #4b5563;
            padding: 0.65rem 0.9rem;
            margin: 0.5rem 0 1rem 0;
            background: rgba(127, 127, 127, 0.08);
            border-radius: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="hasamex-header">
            <h1>🔎 {settings.app_name}</h1>
            <div class="hasamex-subtitle">
                Grounded expert-transcript analysis with
                timestamped source evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page(
    *,
    navigation: str,
    services: ApplicationServices,
) -> None:
    """
    Route the selected navigation item to its UI renderer.
    """
    renderers: dict[
        str,
        Callable[[], None],
    ] = {
        "Dashboard": lambda: render_dashboard(
            services.retriever,
            services.vector_store,
        ),
        "Interview Guide": lambda: render_interview_guide(
            services.interview_guide,
            services.retriever,
        ),
        "Themes": lambda: render_themes(
            services.themes,
        ),
        "Disagreements": lambda: render_disagreements(
            services.disagreements,
        ),
        "Ask the Transcripts": lambda: render_qa(
            services.qa,
        ),
        "Sources": lambda: render_sources(
            services.retriever,
        ),
    }

    renderer = renderers.get(
        navigation
    )

    if renderer is None:
        st.error(
            f"Unknown application section: {navigation}"
        )
        logger.error(
            "Unknown navigation value: %s",
            navigation,
        )
        return

    try:
        renderer()

    except Exception as error:
        _handle_page_error(
            navigation,
            error,
        )


def _handle_initialization_error(
    error: Exception,
) -> None:
    """
    Render a safe application initialization error.
    """
    logger.exception(
        "Application initialization error."
    )

    st.error(
        "The application could not initialize its analysis services."
    )

    st.markdown(
        """
        ### Before continuing

        Make sure that:

        1. The environment variables are configured.
        2. The transcript ingestion pipeline has been executed.
        3. The FAISS vector store has been created.
        4. The configured embedding model is available.
        5. The Google API key is configured for LLM-powered analysis.
        """
    )

    with st.expander(
        "Technical details"
    ):
        st.code(
            str(error)
        )


def _handle_page_error(
    page_name: str,
    error: Exception,
) -> None:
    """
    Handle an exception raised by a page renderer.
    """
    logger.exception(
        "Error while rendering page '%s'.",
        page_name,
    )

    st.error(
        f"The '{page_name}' section encountered an error."
    )

    st.caption(
        "The application logged the detailed exception. "
        "Review the application log for debugging information."
    )

    with st.expander(
        "Technical details"
    ):
        st.code(
            str(error)
        )


if __name__ == "__main__":
    main()