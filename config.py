import os

from dotenv import load_dotenv

# Load env vars from .env before any os.getenv calls
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CLAUDE_MODEL = "claude-opus-4-8"  # model used for final answer generation
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "./chroma_db"  # directory for persisted vector store
CHROMA_COLLECTION_NAME = "regurag_docs"  # collection name inside Chroma
DATA_DIR = "./data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RELEVANCE_THRESHOLD = 0.3
TOP_K = 4

FALLBACK_MESSAGE = (
    "No relevant provisions were found in the current regulatory library.\n"
    "Please consult a qualified compliance professional or refer to the official sources below:\n"
    "  - People's Bank of China: https://www.pbc.gov.cn\n"
    "  - National Financial Regulatory Administration: https://www.nfra.gov.cn\n"
    "  - China Securities Regulatory Commission: https://www.csrc.gov.cn\n"
    "  - SEC EDGAR (US): https://www.sec.gov/edgar"
)
