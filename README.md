# EU AI Act RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed to query, analyze, and navigate the regulatory text of the EU Artificial Intelligence Act.

## Project Structure

```plaintext
eu-ai-act-rag/
├── app/
│   ├── api/          # FastAPI backend routes & endpoints
│   ├── core/         # Core application logic & config
│   └── ui/           # User interface
├── configs/          # Configuration files
├── data/
│   ├── processed/    # Cleaned, chunked, and indexed data
│   └── raw/          # Raw legal and regulatory texts
├── evaluation/       # RAG evaluation metrics & benchmarks
├── ingestion/        # Document ingestion and parsing pipelines
├── notebooks/        # Prototyping and exploratory notebooks
├── retrieval/        # Query engine, retrievers & rerankers
├── scripts/          # Utility scripts
├── tests/            # Unit and integration test suites
├── .env              # Local environment variables (ignored by git)
├── .env.example      # Example environment variables template
├── .gitignore        # Git ignore file
├── README.md         # Project documentation
└── requirements.txt  # Project dependencies
```

## Quickstart

### 1. Set Up Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Environment

```bash
python -c "import dotenv; import llama_index.core; print('Environment is ready!')"
```
