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
- Document loading: LangChain TextLoader, PyPDFLoader, pypdf (parsing dependency)
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
│   └── eval/              # evaluation CSVs
└── src/
    └── regurag/
        ├── app.py         # Streamlit UI
        ├── config.py
        ├── document_loader.py
        ├── evaluate.py
        ├── main.py        # CLI entry point
        ├── qa_chain.py
        ├── smoke/         # small manual checks
        ├── vector_store.py
        └── parser/
```

Generated/local files such as `.env`, `chroma_db/`, `eval_cache.json`, `optimization_log.csv`, virtual environments, and
IDE files are ignored.

## QuickStart

### Run with Docker

```bash
cp .env.example .env
# add your api keys in .env
# place your `.pdf` or `.txt` regulatory files under `data/`

docker build -t regurag .
# (docker run attempts to build ChromaDB from files in data/ before starting the web UI)
docker run --env-file .env -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/chroma_db:/app/chroma_db" \
  regurag
# Open `http://localhost:8501` , and submit your questions!


```

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your api keys in .env

# Place your `.pdf` or `.txt` regulatory files under `data/`
PYTHONPATH=src python -m regurag.main build # chunks data and embeds each chunk, store the resulting vectors into `chroma_db/`

# Option1: ask from the CLI
PYTHONPATH=src python -m regurag.main ask

# Option2: ask from the Web UI
PYTHONPATH=src streamlit run src/regurag/app.py
# Open `http://localhost:8501` , and submit your questions!

```

### Note

- The project does not store embedding model. The first time you run the project, it may take longer because it
  downloads the embedding model from Hugging Face.
- After that, the model is stored locally, so future runs are faster and can reuse the cached copy.

## RAG Evaluation

### Evaluate with Docker

```bash
# Build the image, then build chroma_db before evaluation if needed.
# Evaluation will fail if chroma_db has not been built yet.
docker build -t regurag .
docker run --env-file .env \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/chroma_db:/app/chroma_db" \
  regurag \
  python -m regurag.main build

mkdir -p eval-results

# Run the default small evaluation set:
# Outputs are copied to eval-results/.
docker run --env-file .env \
  -v "$PWD/chroma_db:/app/chroma_db" \
  -v "$PWD/tests/eval:/app/tests/eval:ro" \
  -v "$PWD/eval-results:/outputs" \
  regurag \
  sh -c "python -m regurag.evaluate && cp eval_cache.json optimization_log.csv /outputs/"

# Optional: use a custom evaluation csv
docker run --env-file .env \
  -v "$PWD/chroma_db:/app/chroma_db" \
  -v "$PWD/tests/eval:/app/tests/eval:ro" \
  -v "$PWD/eval-results:/outputs" \
  regurag \
  sh -c "python -m regurag.evaluate --csv tests/eval/test_set.csv && cp eval_cache.json optimization_log.csv /outputs/"
```

### Evaluate locally

```bash
# Install evaluation-only dependencies:
pip install -r requirements-eval.txt

# Evaluation requires both ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env.

# Run the default small evaluation set:
PYTHONPATH=src python -m regurag.evaluate

# Optional: Use a custom evaluation csv
# custom csv columns: id,question,gold_answer,source_clause,type,notes
# Supported `type` values:
#- `normal`: the document library should contain a direct answer.
#- `hard`: the answer exists but retrieval or wording may be harder.
#- `fallback`: the document library should not answer the question.
PYTHONPATH=src python -m regurag.evaluate --csv path/to/test_set.csv

# The evaluation results will show in command line and also be saved into `optimization_log.csv`

```

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
