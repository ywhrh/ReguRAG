import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.language.ChineseSplitter import ChineseSplitter


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
            # UTF-8 support non-English
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
    两阶段切分：先按"第X条"边界切，再用 RecursiveCharacterTextSplitter 处理超长条款。

    阶段1：ChineseSplitter — 保证每个 chunk 对应一个完整条款，不跨条款混入上下文。
    阶段2：RecursiveCharacterTextSplitter — 对仍然超过 CHUNK_SIZE 的条款二次切分。
    """
    # 阶段1：按条款边界切分
    article_chunks = ChineseSplitter.split_documents(documents)
    print(f"  阶段1（按条款切分）: {len(documents)} segment(s) → {len(article_chunks)} 条款")

    # 阶段2：对超长条款再次切分
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
