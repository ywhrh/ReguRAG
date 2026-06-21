import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the local sentence-transformers multilingual embedding model.

    Why a local model instead of an OpenAI/Anthropic embedding API?
    - Zero API cost: indexing thousands of chunks via a paid API gets expensive fast.
    - Works offline: no internet dependency after the first download.
    - paraphrase-multilingual-MiniLM-L12-v2 handles both Chinese and English well.

    normalize_embeddings=True: normalizes vectors to unit length so that
    cosine similarity equals dot product, making Chroma relevance scores intuitive.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~400 MB from HuggingFace; subsequent runs use the local cache)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},             # use CPU when no GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model ready\n")
    return embeddings


def build_vector_store(chunks: list) -> Chroma:
    """
    Embed all chunks and write them to a Chroma vector store.

    What Chroma.from_documents does internally:
    1. Calls the embedding model on each chunk's page_content to produce a vector.
    2. Stores (vector, raw text, metadata) together in a local SQLite file.
    3. Persists to CHROMA_PERSIST_DIR so the next startup can load without rebuilding.

    collection_metadata sets the distance metric to cosine similarity so that
    relevance scores are on a 0–1 scale (1 = perfect match, 0 = unrelated).
    """
    embeddings = get_embedding_model()

    print(f"Embedding and indexing {len(chunks)} chunk(s) into Chroma...")
    print("This may take a few minutes for large document sets — progress is not shown per chunk.\n")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # cosine distance metric
    )

    count = vector_store._collection.count()
    print(f"Vector store built: {count} record(s) persisted to {CHROMA_PERSIST_DIR}/")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing Chroma vector store from disk (no rebuild).
    Requires 'python main.py build' to have been run first.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector store directory '{CHROMA_PERSIST_DIR}' not found.\n"
            "Run: python main.py build"
        )

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    count = vector_store._collection.count()
    print(f"Vector store loaded: {count} record(s) from {CHROMA_PERSIST_DIR}/")
    return vector_store


def retrieve_relevant_chunks(vector_store: Chroma, query: str) -> list:
    """
    Search the vector store for the most relevant chunks for a given query.

    Returns: [(Document, score), ...] sorted by descending relevance.
    Scores are in the range 0–1: closer to 1 means more relevant.
    """
    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=TOP_K,
    )
    return results
