# ReguRAG

ReguRAG is a retrieval-augmented generation system for financial regulatory Q&A. It loads local regulatory documents,
builds a Chroma vector index, retrieves relevant passages for each question, and asks Claude to answer only from those
passages.

The project is designed for compliance-style workflows where traceability matters. Answers include source passages, and
low-relevance questions are blocked before the LLM call to reduce unsupported answers.

## Features

- Source-grounded answers with retrieved passage citations.
- Relevance threshold guardrail before answer generation.
- Local multilingual embeddings with `sentence-transformers`.
- Chroma persistence for local development.
- Streamlit UI for manual testing.
- RAGAS-based evaluation script for retrieval and answer quality checks.

## Stack / AI Model

RAG

- Text splitting: custom Chinese article splitter + LangChain RecursiveCharacterTextSplitter
- Document loading: LangChain TextLoader, PyPDFLoader, pydf (parsing dependency)
- RAG framework: LangChain
- Vector database: ChromaDB
- Embedding model: paraphrase-multilingual-MiniLM-L12-v2
- LLM provider: Anthropic Claude, claude-sonnet-4-6 for answer generation

Evaluation

- framework: RAGAS
- Evaluation LLM: claude-haiku-4-5-20251001
- Evaluation embeddings: Voyage AI voyage-3

UI

- Streamlit

Config & Packaging

- Config / secrets: .env with python-dotenv
- Packaging: pyproject.toml / setuptools
- Containerization: Docker

## Project Layout

```text
ReguRAG/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-eval.txt
├── README.md
├── .env.example
├── data/                  # local regulatory source files
├── tests/
│   ├── eval/              # evaluation CSVs
│   └── smoke/             # small manual checks
└── src/
    └── regurag/
        ├── app.py         # Streamlit UI
        ├── config.py
        ├── document_loader.py
        ├── evaluate.py
        ├── main.py        # CLI entry point
        ├── qa_chain.py
        ├── vector_store.py
        └── parser/
```

Generated/local files such as `.env`, `chroma_db/`, `eval_cache.json`, `optimization_log.csv`, virtual environments, and
IDE files are ignored.

## QuickStart

clone the repo

create a local .env file, with your ANTHROPIC_API_KEY and VOYAGE_API_KEY

```bash
docker build -t regurag .
docker run --env-file .env -p 8501:8501 \
  -v "$PWD/chroma_db:/app/chroma_db" \
  regurag
```

Open `http://localhost:8501` , and submit your questions!

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Then add:

```text
ANTHROPIC_API_KEY=your_key_here
```

The embedding model downloads from Hugging Face on first use and is cached locally afterward.

## Build The Index

Place `.pdf` or `.txt` regulatory files under `data/`, then run:

```bash
PYTHONPATH=src python -m regurag.main build
```

This creates `chroma_db/`.

## Ask From The CLI

```bash
PYTHONPATH=src python -m regurag.main ask
```

## Run The Web UI

```bash
PYTHONPATH=src streamlit run src/regurag/app.py
```

Open `http://localhost:8501`.

## Docker

Build the index on the host first so `chroma_db/` exists, then build and run the UI container:

```bash
docker build -t regurag .
docker run --env-file .env -p 8501:8501 \
  -v "$PWD/chroma_db:/app/chroma_db" \
  regurag
```

Open `http://localhost:8501`.

## Evaluation

Install evaluation-only dependencies:

```bash
pip install -r requirements-eval.txt
```

Run the default small evaluation set:

```bash
PYTHONPATH=src python -m regurag.evaluate
```

Use a custom CSV:

```bash
PYTHONPATH=src python -m regurag.evaluate --csv path/to/test_set.csv
```

Expected CSV columns:

```text
id,question,gold_answer,source_clause,type,notes
```

Supported `type` values:

- `normal`: the document library should contain a direct answer.
- `hard`: the answer exists but retrieval or wording may be harder.
- `fallback`: the document library should not answer the question.

## Configuration

Core settings live in `src/regurag/config.py`.

| Setting               | Default                                 | Purpose                                       |
|-----------------------|-----------------------------------------|-----------------------------------------------|
| `CLAUDE_MODEL`        | `claude-sonnet-4-6`                     | Final answer generation model                 |
| `EMBEDDING_MODEL`     | `paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model                         |
| `CHUNK_SIZE`          | `500`                                   | Maximum chunk size                            |
| `CHUNK_OVERLAP`       | `100`                                   | Chunk overlap                                 |
| `TOP_K`               | `4`                                     | Number of retrieved candidates                |
| `RELEVANCE_THRESHOLD` | `0.6`                                   | Minimum score required before calling the LLM |

## Notes

This project is for development and demonstration. It is not legal, regulatory, or compliance advice. Production use
would need stronger document versioning, jurisdiction metadata, retrieval evaluation, audit logging, and human review.
