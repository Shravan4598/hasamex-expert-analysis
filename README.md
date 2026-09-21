# Hasamex Expert Analysis

A production-oriented RAG application for analyzing expert interview transcripts with **traceable answers, exact quotes, timestamps, cross-expert themes, disagreements, and evidence-grounded Q&A**.

Built for the **Hasamex AI Engineer Technical Case Study**.

---

## 1. Overview

Hasamex Expert Analysis analyzes multiple expert-call transcripts from the same project and converts unstructured interview conversations into structured, evidence-backed insights.

The application supports:

* Uploading and processing expert transcripts
* Answering predefined interview-guide questions for each expert
* Extracting exact supporting quotes
* Showing source timestamps
* Verifying generated evidence against the original transcript
* Identifying common themes across experts
* Identifying areas of disagreement
* Asking natural-language questions across all transcripts
* Returning insufficient-evidence responses when the available evidence does not support an answer

The core design principle is:

> **Every important generated insight should be traceable back to transcript evidence.**

---

## 2. Problem Statement

The case study provides three expert interviews discussing robotic-surgery adoption.

The goal is to build an application that can:

1. Read all three transcripts.
2. Answer the interview-guide questions for every expert.
3. Provide exact supporting quotes.
4. Provide source timestamps.
5. Identify common themes.
6. Identify disagreements.
7. Allow users to ask questions across all transcripts.

The system is designed so that generated answers are grounded in retrieved transcript evidence rather than relying only on the language model's internal knowledge.

---

## 3. Expert Interviews

The demonstration dataset contains three expert interviews:

| Expert           | Market  | Background                           |
| ---------------- | ------- | ------------------------------------ |
| Dr. Jean Martin  | France  | Head of Urology                      |
| Anna Keller      | Germany | Former Hospital Procurement Director |
| Dr. Emily Carter | UK      | Consultant Urologist                 |

The interview guide contains six questions covering:

1. Current robotic-surgery adoption
2. Main adoption barriers
3. Hospital budgets and ROI
4. Surgeon training and clinical outcomes
5. Expected 3–5 year adoption trends
6. Hospital purchasing decision timelines

---

## 4. Key Features

### Interview Guide Analysis

The application automatically analyzes every expert against the six predefined interview questions.

Each answer can contain:

* Expert
* Market
* Question
* Answer
* Supporting evidence
* Exact quote
* Timestamp
* Source transcript
* Evidence status

### Exact Quote Extraction

The system attempts to return the original wording from the transcript instead of generating a paraphrased quotation.

### Timestamp Citations

Evidence is linked to the original transcript timestamp so users can quickly locate the source statement.

Example:

```text
[02:18]
"ROI is very important..."
```

### Cross-Transcript Q&A

Users can ask questions across all available expert interviews.

The system retrieves relevant evidence before generating an answer.

### Common Themes

The application identifies recurring topics discussed by multiple experts, such as:

* Capital budgets
* ROI
* Procedure volume
* Training
* Clinical outcomes
* Procurement timelines
* Gradual adoption

### Disagreement Analysis

The system identifies meaningful differences between expert perspectives rather than assuming that all experts share the same view.

### Insufficient-Evidence Handling

When retrieved evidence does not adequately support an answer, the application can return an insufficient-evidence response instead of fabricating information.

---

# 5. Architecture

```text
                    ┌──────────────────────┐
                    │      Streamlit UI    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Application Layer  │
                    │ Interview / Themes /  │
                    │ Disagreements / Q&A   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Evidence Retrieval   │
                    │   + Reranking         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FAISS Vector Store  │
                    │ + Embeddings          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Transcript Chunks     │
                    │ + Metadata            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Transcript Ingestion  │
                    │ Loading / Parsing /   │
                    │ Chunking              │
                    └──────────────────────┘
```

For generation, the application uses Google's Gemini model through the `google-genai` SDK.

---

# 6. RAG Pipeline

The application follows a retrieval-augmented generation architecture.

```text
Transcript
    │
    ▼
Load
    │
    ▼
Parse
    │
    ▼
Timestamp-aware chunks
    │
    ▼
Sentence Transformer embeddings
    │
    ▼
FAISS vector index
    │
    ▼
Semantic retrieval
    │
    ▼
Reranking
    │
    ▼
Evidence verification
    │
    ▼
Gemini generation
    │
    ▼
Answer + Quote + Timestamp + Source
```

The language model is therefore provided with relevant transcript evidence rather than being asked to answer solely from its pretrained knowledge.

---

# 7. Transcript Ingestion

The ingestion pipeline is separated into multiple stages.

### Loader

Responsible for reading transcript files and creating the initial document representation.

### Parser

Extracts transcript structure, including:

* Speaker information
* Timestamp information
* Transcript text
* Segment boundaries
* Source metadata

### Chunker

Splits transcript content into retrieval-friendly chunks while preserving relevant metadata.

Chunk metadata can include:

* Transcript identifier
* Expert
* Market
* Timestamp
* Speaker
* Chunk index
* Original text

Preserving this metadata is important because retrieval must eventually produce traceable evidence.

---

# 8. Embeddings

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings convert transcript chunks into numerical vectors that can be compared for semantic similarity.

This allows a question such as:

```text
How important is ROI?
```

to retrieve transcript statements discussing:

```text
payback period
procedure volume
maintenance cost
economic case
financial approval
```

even when the exact words do not match.

---

# 9. FAISS Retrieval

FAISS is used as the local vector-search engine.

The retrieval pipeline:

1. Embeds the user's query.
2. Searches the FAISS index.
3. Retrieves the highest-scoring candidate chunks.
4. Applies additional ranking/filtering.
5. Passes the strongest evidence to the analysis layer.

The default configuration includes:

```text
RETRIEVAL_TOP_K=8
RERANK_TOP_K=5
```

This keeps the generation context focused on the most relevant evidence.

---

# 10. Reranking

Initial vector retrieval provides candidate evidence.

A reranking stage is then used to improve the ordering of retrieved evidence before it reaches the generation layer.

This separates:

```text
Candidate Retrieval
        ↓
Evidence Ranking
        ↓
Generation
```

rather than directly generating an answer from every retrieved chunk.

---

# 11. Evidence Verification

Evidence verification is a core reliability component.

The application checks whether generated evidence can be matched against the original transcript.

The system distinguishes evidence states such as verified or insufficient evidence.

The purpose is to prevent generated text from being presented as an exact transcript quotation when that wording does not actually exist in the source.

---

# 12. Quotes and Timestamps

Every important answer should retain a connection to its source.

A typical evidence record contains information such as:

```text
Expert
Market
Transcript
Timestamp
Quote
Evidence status
Retrieval score
```

For example:

```text
Expert: Dr. Jean Martin
Market: France
Timestamp: 02:18

Quote:
"ROI is very important..."
```

This allows the user to move from:

```text
Insight
   ↓
Answer
   ↓
Evidence
   ↓
Original transcript location
```

---

# 13. Hallucination Mitigation

The application uses several mechanisms to reduce unsupported answers.

### Retrieval grounding

The model receives retrieved transcript evidence as context.

### Evidence verification

Quotes are checked against the source transcript.

### Source metadata

Retrieved chunks preserve transcript and timestamp metadata.

### Insufficient-evidence handling

The system can decline to make a definitive claim when the retrieved evidence does not adequately support the requested answer.

### Low-temperature generation

The default LLM temperature is:

```text
0.1
```

This favors more deterministic responses for an evidence-analysis workflow.

---

# 14. Interview Guide Analysis

The interview-guide module processes the six predefined questions against the expert evidence.

The questions cover:

```text
1. Current adoption
2. Adoption barriers
3. Budgets and ROI
4. Training and clinical outcomes
5. 3–5 year adoption trend
6. Purchasing decision timeline
```

The system keeps the analysis structured so that users can compare the same question across different experts.

---

# 15. Themes

The themes module analyzes evidence across the interviews to identify recurring topics.

Examples from the demonstration dataset include:

### Capital and Budget

France and Germany both emphasize capital approval and economic justification.

### ROI and Economics

Experts discuss:

* Procedure volume
* Utilisation
* Maintenance
* Payback
* Total cost of ownership
* Financial approval

### Training

Training capacity is repeatedly identified as an important factor in successful adoption.

### Clinical Outcomes

Experts recognize clinical outcomes as important while differing in how strongly they balance clinical considerations against economic considerations.

### Gradual Adoption

The interviews generally describe continued growth rather than an immediate universal transition.

---

# 16. Disagreements

The application also identifies differences in expert perspectives.

Examples include differences in:

* Strength of expected adoption growth
* Relative importance of economics versus clinical considerations
* Procurement timelines
* Funding and training constraints

The system presents these as evidence-backed differences rather than forcing all expert responses into a single conclusion.

---

# 17. Cross-Transcript Q&A

Users can ask questions across all available interviews.

Example:

```text
What are the main barriers to robotic surgery adoption?
```

The application:

```text
Question
   ↓
Retrieve evidence
   ↓
Rank evidence
   ↓
Generate grounded response
   ↓
Attach sources
```

This enables cross-market analysis without requiring users to manually search every transcript.

---

# 18. Technology Stack

| Component        | Technology                        |
| ---------------- | --------------------------------- |
| UI               | Streamlit                         |
| Language Model   | Google Gemini                     |
| LLM SDK          | google-genai                      |
| Embeddings       | Sentence Transformers             |
| Vector Search    | FAISS                             |
| Data Validation  | Pydantic                          |
| Data Processing  | Python / NumPy / Pandas           |
| Testing          | Pytest                            |
| Coverage         | pytest-cov                        |
| Linting          | Ruff                              |
| Containerization | Docker                            |
| Configuration    | python-dotenv / pydantic-settings |

---

# 19. Project Structure

```text
hasamex-expert-analysis/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── exception.py
├── logger.py
├── README.md
│
├── src/
│   ├── analysis/
│   │   ├── disagreements.py
│   │   ├── interview_guide.py
│   │   ├── llm.py
│   │   ├── qa.py
│   │   └── themes.py
│   │
│   ├── evidence/
│   │   ├── citation.py
│   │   └── quote_verifier.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── loader.py
│   │   └── parser.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── ui/
│   │   ├── dashboard.py
│   │   ├── disagreements.py
│   │   ├── interview_guide.py
│   │   ├── qa.py
│   │   ├── sources.py
│   │   └── themes.py
│   │
│   ├── config.py
│   └── models.py
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   └── README.md
│
├── scripts/
│   ├── evaluate.py
│   └── ingest.py
│
├── tests/
│   ├── test_insufficient_evidence.py
│   ├── test_models.py
│   ├── test_parser.py
│   ├── test_quotes.py
│   ├── test_retrieval.py
│   └── test_timestamps.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── storage/
    └── vector_store/
```

---

# 20. Installation

## Clone the repository

```bash
git clone <your-repository-url>
cd hasamex-expert-analysis
```

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 21. Environment Configuration

Create a `.env` file from `.env.example`.

```dotenv
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=2048

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RETRIEVAL_TOP_K=8
RERANK_TOP_K=5

CHUNK_SIZE=1200
CHUNK_OVERLAP=200

MIN_RETRIEVAL_SCORE=0.25
MIN_EVIDENCE_COVERAGE=0.50
```

Never commit the real `.env` file or API keys.

---

# 22. Running Locally

Start the application with:

```bash
python -m streamlit run app.py
```

The application is available at:

```text
http://localhost:8501
```

---

# 23. Docker

The project includes a Dockerfile and Docker Compose configuration.

Build the image:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

The application is available at:

```text
http://localhost:8501
```

Run in detached mode:

```bash
docker compose up -d
```

Stop the application:

```bash
docker compose down
```

View logs:

```bash
docker compose logs --tail=100
```

---

# 24. Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
pytest -q
```

Current validation:

```text
143 passed
```

Run linting:

```bash
ruff check .
```

Current validation:

```text
All checks passed!
```

Run coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

The current automated suite focuses heavily on:

* Data models
* Transcript parsing
* Timestamp handling
* Quote verification
* Retrieval behavior
* Insufficient-evidence behavior

The Streamlit presentation layer was additionally validated through manual UI testing.

---

# 25. Evaluation

The evaluation framework is designed to assess whether generated analysis is supported by expected evidence.

Evaluation data is maintained under:

```text
evaluation/
```

The evaluation pipeline can be run using:

```bash
python scripts/evaluate.py
```

The evaluation approach focuses on evidence quality and answer grounding rather than simply measuring whether an LLM-generated response sounds plausible.

---

# 26. Reliability Design

The application separates several responsibilities:

```text
Ingestion
    ↓
Parsing
    ↓
Chunking
    ↓
Embedding
    ↓
Retrieval
    ↓
Reranking
    ↓
Evidence Verification
    ↓
Analysis
    ↓
Presentation
```

This separation makes individual components easier to test, replace, and scale.

For example:

* The embedding model can be changed independently.
* The vector database can be replaced later.
* The LLM provider is isolated in the analysis layer.
* UI components are separated from retrieval logic.
* Evidence verification is independent of generation.

---

# 27. Scalability: 3 → 30+ Transcripts

The current case study contains three transcripts, but the architecture is designed to support a larger corpus.

For a larger workload:

### Current local architecture

```text
3 transcripts
     ↓
Local parsing
     ↓
Local embeddings
     ↓
FAISS
     ↓
RAG
```

### Larger production architecture

```text
30+ transcripts
       ↓
Object/document storage
       ↓
Asynchronous ingestion
       ↓
Chunking + metadata extraction
       ↓
Embedding workers
       ↓
Persistent vector database
       ↓
Metadata filtering
       ↓
Reranking
       ↓
LLM generation
```

For a larger deployment, additional improvements could include:

* Persistent vector storage
* Batch embedding
* Background ingestion jobs
* Metadata-based filtering
* Caching
* Query-result caching
* Parallel processing
* Observability and tracing
* Rate limiting
* Authentication and authorization
* Dedicated API/backend service
* Cloud object storage
* Managed vector database

The important design principle is that ingestion and retrieval are separated from the UI, allowing the processing layer to scale independently.

---

# 28. Security Considerations

The project should not commit secrets.

Sensitive runtime configuration is supplied through environment variables.

The `.gitignore` excludes:

```text
.env
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
logs/
```

API keys should never be hard-coded into Python source code, Dockerfiles, or Compose files.

For a production deployment, additional controls would include:

* Authentication
* Authorization
* Secret management
* TLS
* Audit logging
* Request limits
* Secure transcript storage
* Data retention policies

---

# 29. Limitations

The current implementation is designed for the Hasamex technical case study and local/demo deployment.

Current limitations include:

* Local FAISS-based vector storage
* Local transcript processing
* Dependency on the configured Gemini API
* Streamlit as the application interface
* Limited demonstration corpus
* No production authentication layer
* No distributed ingestion infrastructure

These limitations are deliberate for the scope of the technical case study and can be addressed in a production deployment.

---

# 30. Future Improvements

Potential production enhancements include:

1. Persistent cloud vector storage
2. Background document ingestion
3. Larger-scale batch processing
4. Authentication and authorization
5. API-based backend architecture
6. Observability and distributed tracing
7. Evaluation dashboards
8. Automated regression evaluation
9. Human feedback loops
10. More advanced multilingual retrieval
11. Document-level access controls
12. Query and embedding caching

---

# 31. Quality Validation

The project has been validated through multiple layers:

```text
Static Analysis
      ↓
Ruff
      ↓
Automated Tests
      ↓
143 tests passing
      ↓
Manual Streamlit UI Testing
      ↓
Docker Build
      ↓
Docker Runtime Testing
```

Current known validation status:

```text
Ruff                    PASS
Automated tests         143 PASS
Local Streamlit         PASS
Docker build            PASS
Docker startup          PASS
Docker Streamlit        PASS
```

---

# 32. Case Study Design Decisions

### Why RAG?

The task requires answers to be grounded in specific transcripts. RAG provides a mechanism to retrieve relevant source evidence before generation.

### Why FAISS?

FAISS provides efficient local vector similarity search and is suitable for the relatively small case-study corpus.

### Why Gemini?

Gemini provides the language-generation capability required to synthesize retrieved transcript evidence into structured answers.

### Why sentence-transformer embeddings?

They provide local semantic embeddings without requiring an external embedding API for every retrieval operation.

### Why evidence verification?

Retrieval alone does not guarantee that a generated quotation exactly matches the source. Verification provides an additional reliability layer.

### Why timestamps?

Timestamps allow users to trace an answer back to the original conversation.

---

# 33. Interview Demonstration Flow

A concise demonstration can follow this sequence:

```text
1. Show the three expert transcripts
2. Show the Interview Guide
3. Select an interview question
4. Show expert-specific answers
5. Open supporting evidence
6. Show exact quote + timestamp
7. Show common themes
8. Show disagreements
9. Ask a cross-transcript question
10. Demonstrate insufficient evidence
11. Explain the RAG architecture
12. Explain scaling from 3 → 30+ transcripts
```

The most important demonstration principle is to show that the application does not merely produce an answer—it also shows **why the answer can be trusted and where the evidence came from**.

---

# 34. License

This project is prepared as a technical case-study submission for Hasamex.

---

## Author

**Shravan Kumar Pandey**

B.Tech — Computer Science Engineering
Specialization: Data Science
