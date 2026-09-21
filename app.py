"""
Hasamex Expert Analysis - Streamlit Application

This module provides the user-facing Streamlit application for:
    - Expert interview analysis
    - Theme analysis
    - Disagreement analysis
    - Interview guide generation
    - Source-grounded transcript Q&A

The application layer is responsible for UI orchestration only.
Business logic remains inside the analysis and retrieval modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from exception import SensorException
from logger import logging
from src.analysis.disagreements import DisagreementAnalyzer
from src.analysis.interview_guide import InterviewGuideAnalyzer
from src.analysis.llm import GeminiLLMService
from src.analysis.qa import TranscriptQA
from src.analysis.themes import ThemeAnalyzer
from src.config import Settings, get_settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.reranker import RetrievalReranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class ApplicationInitializationError(Exception):
    """Raised when application services cannot be initialized."""


@dataclass
class ApplicationServices:
    """Container for all application-level services."""

    settings: Settings
    embeddings: EmbeddingService
    vector_store: FAISSVectorStore
    retriever: Retriever
    reranker: RetrievalReranker
    llm: GeminiLLMService
    interview_guide: InterviewGuideAnalyzer
    themes: ThemeAnalyzer
    disagreements: DisagreementAnalyzer
    qa: TranscriptQA


@st.cache_resource
def initialize_services() -> ApplicationServices:
    """
    Initialize and cache all application services.

    Streamlit reruns the application frequently, so expensive resources
    such as embedding models and vector stores should be initialized only
    once per process/session cache.
    """
    try:
        # ------------------------------------------------------------------
        # Settings
        # ------------------------------------------------------------------
        settings = get_settings()
        settings.ensure_directories()

        # ------------------------------------------------------------------
        # Embedding service
        # ------------------------------------------------------------------
        embeddings = EmbeddingService(
            settings=settings,
        )

        # ------------------------------------------------------------------
        # FAISS vector store
        # ------------------------------------------------------------------
        vector_store = FAISSVectorStore(
            embedding_service=embeddings,
            settings=settings,
            storage_dir=settings.vector_store_path,
        )

        if not vector_store.exists_on_disk():
            raise ApplicationInitializationError(
                "The vector store has not been built yet. "
                "Run the ingestion/indexing pipeline before starting "
                "the application."
            )

        vector_store.load()

        # ------------------------------------------------------------------
        # Retriever
        # ------------------------------------------------------------------
        retriever = Retriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            settings=settings,
            top_k=settings.retrieval_top_k,
        )

        # ------------------------------------------------------------------
        # Reranker
        # ------------------------------------------------------------------
        reranker = RetrievalReranker(
            settings=settings,
        )

        # ------------------------------------------------------------------
        # LLM service
        # ------------------------------------------------------------------
        llm = GeminiLLMService(
            settings=settings,
        )

        # ------------------------------------------------------------------
        # Analysis services
        # ------------------------------------------------------------------
        interview_guide = InterviewGuideAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm,
            settings=settings,
        )

        themes = ThemeAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm,
            settings=settings,
        )

        disagreements = DisagreementAnalyzer(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm,
            settings=settings,
        )

        qa = TranscriptQA(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm,
            settings=settings,
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
            "Application service initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error


def display_initialization_error(
    error: Exception,
) -> None:
    """Display a useful initialization error in the Streamlit UI."""

    st.error(
        "The application could not initialize its analysis services."
    )

    st.markdown(
        """
        Please check the technical details below.

        Common causes include:
        - Missing or invalid environment variables
        - Missing FAISS vector-store files
        - Embedding model loading failure
        - Invalid API configuration
        - Dependency/version mismatch
        """
    )

    with st.expander(
        "Technical details",
        expanded=True,
    ):
        st.code(
            f"Exception type: {type(error).__name__}\n\n"
            f"Error message: {error}"
        )


def render_sidebar(
    services: ApplicationServices,
) -> None:
    """Render application sidebar."""

    settings = services.settings

    with st.sidebar:
        st.title("Hasamex Expert Analysis")

        st.markdown(
            """
            **Source-grounded expert transcript analysis**

            Analyze expert interviews and generate answers
            grounded in the indexed transcript evidence.
            """
        )

        st.divider()

        st.subheader("System Status")

        st.success("Services initialized")

        st.write(
            f"**Embedding model:** "
            f"`{settings.embedding_model}`"
        )

        st.write(
            f"**LLM model:** "
            f"`{settings.llm_model}`"
        )

        st.write(
            f"**Retrieval Top-K:** "
            f"`{settings.retrieval_top_k}`"
        )

        st.write(
            f"**Reranking Top-K:** "
            f"`{settings.rerank_top_k}`"
        )

        st.divider()

        if st.button(
            "Clear cached services",
            use_container_width=True,
        ):
            st.cache_resource.clear()
            st.rerun()


def render_home() -> None:
    """Render the application home page."""

    st.title("Hasamex Expert Analysis")

    st.markdown(
        """
        Welcome to the **Hasamex Expert Analysis** application.

        This application provides source-grounded analysis over
        expert interview transcripts.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Available Analysis")

        st.markdown(
            """
            - **Interview Guide**
            - **Themes**
            - **Disagreements**
            - **Transcript Q&A**
            """
        )

    with col2:
        st.subheader("Grounded Evidence")

        st.markdown(
            """
            Answers are generated from retrieved transcript
            evidence and retain source metadata such as:

            - Expert
            - Market
            - Source file
            - Timestamp
            - Retrieval score
            """
        )


def render_qa_page(
    services: ApplicationServices,
) -> None:
    """Render transcript Q&A interface."""

    st.header("Transcript Q&A")

    st.write(
        "Ask a question about the indexed expert transcripts."
    )

    query = st.text_area(
        "Your question",
        placeholder=(
            "Example: What are the key drivers of market adoption?"
        ),
        height=120,
    )

    if st.button(
        "Ask",
        type="primary",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning(
                "Please enter a question."
            )
            return

        try:
            with st.spinner(
                "Searching transcript evidence and generating answer..."
            ):
                response = services.qa.ask(
                    question=query.strip(),
                )

            st.subheader("Answer")

            # GroundedAnswer is a Pydantic model.
            st.write(response.answer)

            # --------------------------------------------------------------
            # Confidence / grounding information
            # --------------------------------------------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Confidence",
                    f"{response.confidence:.2f}",
                )

            with col2:
                st.metric(
                    "Evidence Coverage",
                    f"{response.evidence_coverage:.2%}",
                )

            with col3:
                st.metric(
                    "Evidence Items",
                    len(response.evidence),
                )

            if response.refusal_reason:
                st.warning(
                    f"Grounding note: {response.refusal_reason}"
                )

            # --------------------------------------------------------------
            # Evidence
            # --------------------------------------------------------------
            if response.evidence:
                st.subheader("Source Evidence")

                for index, evidence in enumerate(
                    response.evidence,
                    start=1,
                ):
                    with st.expander(
                        f"Evidence {index}: "
                        f"{evidence.expert_name or 'Unknown Expert'}"
                    ):
                        st.write(
                            f"**Expert:** "
                            f"{evidence.expert_name or 'Unknown'}"
                        )

                        st.write(
                            f"**Market:** "
                            f"{evidence.market or 'Unknown'}"
                        )

                        st.write(
                            f"**Source:** "
                            f"{evidence.source_file}"
                        )

                        st.write(
                            f"**Timestamp:** "
                            f"{evidence.start_timestamp}"
                        )

                        if evidence.end_timestamp:
                            st.write(
                                f"**End timestamp:** "
                                f"{evidence.end_timestamp}"
                            )

                        st.markdown("**Quote:**")

                        st.info(evidence.quote)

            # --------------------------------------------------------------
            # Citations
            # --------------------------------------------------------------
            if response.citations:
                st.subheader("Citations")

                for citation in response.citations:
                    st.write(citation)

        except SensorException as error:
            logger.exception(
                "Transcript Q&A failed."
            )

            st.error(
                f"Unable to answer the question: {error}"
            )

        except Exception as error:
            logger.exception(
                "Unexpected Transcript Q&A failure."
            )

            st.error(
                f"Unexpected error: {type(error).__name__}: {error}"
            )


def render_theme_page(
    services: ApplicationServices,
) -> None:
    """Render theme analysis interface."""

    st.header("Theme Analysis")

    query = st.text_input(
        "Topic or theme",
        placeholder=(
            "Example: pricing, adoption, competition"
        ),
    )

    if st.button(
        "Analyze Theme",
        type="primary",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning(
                "Please enter a theme."
            )
            return

        try:
            with st.spinner(
                "Analyzing transcript evidence..."
            ):
                result = services.themes.analyze(
                    focus=query.strip()
                )

            st.subheader("Analysis")

            st.write(result)

        except SensorException as error:
            logger.exception(
                "Theme analysis failed."
            )

            st.error(
                f"Theme analysis failed: {error}"
            )

        except Exception as error:
            logger.exception(
                "Unexpected theme analysis failure."
            )

            st.error(
                f"Unexpected error: {type(error).__name__}: {error}"
            )


def render_disagreement_page(
    services: ApplicationServices,
) -> None:
    """Render disagreement analysis interface."""

    st.header("Disagreement Analysis")

    query = st.text_input(
        "Topic to compare",
        placeholder=(
            "Example: Market growth expectations"
        ),
    )

    if st.button(
        "Analyze Disagreements",
        type="primary",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning(
                "Please enter a topic."
            )
            return

        try:
            with st.spinner(
                "Comparing expert evidence..."
            ):
                result = services.disagreements.analyze(
                    focus=query.strip()
                )

            st.subheader("Analysis")

            st.write(result)

        except SensorException as error:
            logger.exception(
                "Disagreement analysis failed."
            )

            st.error(
                f"Disagreement analysis failed: {error}"
            )

        except Exception as error:
            logger.exception(
                "Unexpected disagreement analysis failure."
            )

            st.error(
                f"Unexpected error: {type(error).__name__}: {error}"
            )


def render_interview_guide_page(
    services: ApplicationServices,
) -> None:
    """Render interview guide analysis interface."""

    st.header("Interview Guide")

    topic = st.text_input(
        "Interview topic",
        placeholder=(
            "Example: robotic surgery adoption"
        ),
    )

    if st.button(
        "Generate Interview Guide",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning(
                "Please enter an interview topic."
            )
            return

        try:
            with st.spinner(
                "Analyzing transcript evidence..."
            ):
                result = services.interview_guide.analyze_question(
                    topic.strip()
                )

            st.subheader("Interview Guide")

            st.write(result)

        except SensorException as error:
            logger.exception(
                "Interview guide generation failed."
            )

            st.error(
                f"Interview guide generation failed: {error}"
            )

        except Exception as error:
            logger.exception(
                "Unexpected interview guide failure."
            )

            st.error(
                f"Unexpected error: {type(error).__name__}: {error}"
            )


def main() -> None:
    """Run the Streamlit application."""

    settings = get_settings()

    st.set_page_config(
        page_title=settings.streamlit_page_title,
        page_icon=settings.streamlit_page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        services = initialize_services()

    except Exception as error:  # noqa: BLE001
        display_initialization_error(error)
        st.stop()
        return

    render_sidebar(services)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Transcript Q&A",
            "Theme Analysis",
            "Disagreement Analysis",
            "Interview Guide",
        ],
    )

    if page == "Home":
        render_home()

    elif page == "Transcript Q&A":
        render_qa_page(services)

    elif page == "Theme Analysis":
        render_theme_page(services)

    elif page == "Disagreement Analysis":
        render_disagreement_page(services)

    elif page == "Interview Guide":
        render_interview_guide_page(services)


if __name__ == "__main__":
    main()