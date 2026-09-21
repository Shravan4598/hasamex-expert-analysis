"""
Streamlit UI package for the Hasamex Expert Analysis application.

The UI layer is responsible only for presentation and user interaction.

Core responsibilities such as:

    - transcript ingestion
    - retrieval
    - reranking
    - evidence verification
    - LLM analysis
    - citation generation

remain outside this package.

This separation keeps the application maintainable and allows the
analysis components to be tested independently of Streamlit.
"""

__all__ = [
    "dashboard",
    "disagreements",
    "interview_guide",
    "qa",
    "sources",
    "themes",
]