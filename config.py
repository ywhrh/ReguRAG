import os

from dotenv import load_dotenv

# Load env vars from .env before any os.getenv calls
load_dotenv()

# ──────────────────────────────────────────────
# Anthropic API
# ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CLAUDE_MODEL = "claude-opus-4-8"  # model used for final answer generation

# ──────────────────────────────────────────────
# Local embedding model (multilingual, no API cost)
# Downloaded automatically from HuggingFace on first run (~400 MB), then cached locally
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────
# Document chunking parameters (adjust here when tuning)
# chunk_size:    max characters per chunk
# chunk_overlap: overlap between adjacent chunks to avoid cutting across key sentences
# ──────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHROMA_PERSIST_DIR = "./chroma_db"  # directory for persisted vector store
CHROMA_COLLECTION_NAME = "regurag_docs"  # collection name inside Chroma
DATA_DIR = "./data"
TOP_K = 4  # number of top chunks to retrieve per query

# Anti-hallucination relevance threshold (0–1, higher = stricter)
# If the best retrieved chunk scores below this, the LLM is never called.
# Start at 0.3 and tune up (stricter) or down (more permissive) based on results.
RELEVANCE_THRESHOLD = 0.3

# ──────────────────────────────────────────────
# Fallback message (returned instead of calling the LLM when relevance is too low)
# ──────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "No relevant provisions were found in the current regulatory library.\n"
    "Please consult a qualified compliance professional or refer to the official sources below:\n"
    "  - People's Bank of China: https://www.pbc.gov.cn\n"
    "  - National Financial Regulatory Administration: https://www.nfra.gov.cn\n"
    "  - China Securities Regulatory Commission: https://www.csrc.gov.cn\n"
    "  - SEC EDGAR (US): https://www.sec.gov/edgar"
)
