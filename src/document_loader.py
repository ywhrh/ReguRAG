import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


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

        if filename.endswith(".txt"):
            # UTF-8 encoding is required for Chinese-language documents
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"  [OK] loaded txt: {filename} ({len(docs)} segment(s))")

        elif filename.endswith(".pdf"):
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
    Split documents into smaller chunks suitable for embedding models.

    Why RecursiveCharacterTextSplitter?
    - Tries to split at semantic boundaries first: paragraphs, then newlines,
      then punctuation — rather than cutting at a fixed character count.
    - The separators list is tried in order; the first one that fits is used.
    - chunk_size and chunk_overlap are centrally configured in config.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,            # count by characters (correct for CJK text)
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Splitting complete: {len(documents)} segment(s) → {len(chunks)} chunk(s)")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    return chunks
