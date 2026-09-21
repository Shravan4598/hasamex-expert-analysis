"""
Analysis package for the Hasamex Expert Analysis application.

This package contains the application-level reasoning workflows:

    - Interview-guide question answering
    - Cross-expert theme analysis
    - Cross-expert disagreement/difference analysis
    - Free-form transcript question answering

The package intentionally keeps these workflows separate from:
    - ingestion
    - retrieval
    - evidence verification
    - Streamlit UI

This separation makes the application easier to test, maintain, and
extend from the initial three transcripts to a larger expert corpus.
"""

__all__ = [
    "disagreements",
    "interview_guide",
    "llm",
    "qa",
    "themes",
]