import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from regurag.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from regurag.parser.chinese_splitter import ChineseSplitter

SUPPORTED_EXTENSIONS = (".txt", ".pdf")


def get_supported_document_files(data_dir: str = DATA_DIR) -> list[str]:
    """Return supported source document filenames in the data directory."""
    if not os.path.exists(data_dir):
        return []

    return [
        filename
        for filename in sorted(os.listdir(data_dir))
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    ]


def load_documents(data_dir: str = DATA_DIR) -> list:
    """
    Load all .txt and .pdf regulatory documents from the given directory.

    Returns a list of LangChain Document objects. Each Document carries:
      - page_content: the raw text
      - metadata["source"]: file path, used later for answer citation
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            "Create it and add regulatory documents before running."
        )

    documents = []

    for filename in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, filename)

        if filename.lower().endswith(".txt"):
            # UTF-8 support non-English
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"  [OK] loaded txt: {filename} ({len(docs)} segment(s))")

        elif filename.lower().endswith(".pdf"):
            # PyPDFLoader splits by page; each page becomes one Document
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  [OK] loaded pdf: {filename} ({len(docs)} page(s))")

        else:
            # Skip unsupported formats silently (ignore hidden files)
            if not filename.startswith("."):
                print(f"  [skip] unsupported format: {filename}")

    print(f"\nTotal loaded: {len(documents)} document segment(s)")
    return documents


def split_documents(documents: list) -> list:
    """
    Split in two stages:
    1. Split Chinese regulations by article boundary where possible.
    2. Split long articles again with RecursiveCharacterTextSplitter.
    """
    article_chunks = ChineseSplitter.split_documents(documents)
    print(f"  Article split: {len(documents)} segment(s) -> {len(article_chunks)} article(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(article_chunks)

    print(f"Splitting complete: {len(documents)} segment(s) → {len(chunks)} chunk(s)")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    return chunks
