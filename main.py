"""
ReguRAG — command-line entry point

Usage:
  python main.py build   # load documents from data/, chunk, and build the vector store
  python main.py ask     # start interactive Q&A (requires a built vector store)
"""
import sys
import os

# Add the project root to sys.path so that src/ modules can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ANTHROPIC_API_KEY, DATA_DIR
from src.document_loader import load_documents, split_documents
from src.vector_store import build_vector_store, load_vector_store
from src.qa_chain import ask


def check_api_key():
    """Fail fast if the API key is missing, rather than crashing mid-session."""
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Create a .env file in the project root containing:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        print("Get a key at: https://console.anthropic.com/keys")
        sys.exit(1)


def cmd_build():
    """
    Build command:
    1. Load all .txt / .pdf documents from data/
    2. Split into chunks
    3. Embed and store in Chroma (persisted to chroma_db/)
    """
    print("=" * 55)
    print("  ReguRAG — Build Index")
    print("=" * 55)

    # Step 1: load documents
    print(f"\n[1/3] Loading documents from {DATA_DIR}/")
    documents = load_documents(DATA_DIR)

    if not documents:
        print(f"\nError: no .txt or .pdf files found in {DATA_DIR}/")
        print("Add regulatory documents to the data/ directory and retry.")
        sys.exit(1)

    # Step 2: split into chunks
    print("\n[2/3] Splitting into chunks")
    chunks = split_documents(documents)

    # Step 3: embed and index
    print("\n[3/3] Embedding and writing to Chroma")
    build_vector_store(chunks)

    print("\n" + "=" * 55)
    print("  Index built successfully!")
    print("  Start Q&A with:")
    print("    python main.py ask")
    print("=" * 55)


def cmd_ask():
    """
    Ask command: load the existing vector store and start an interactive Q&A loop.
    Each answer includes citations so users can verify against the original text.
    """
    check_api_key()

    print("=" * 55)
    print("  ReguRAG — Financial Regulatory Q&A")
    print("  Type 'quit' or press Enter on an empty line to exit")
    print("=" * 55 + "\n")

    # Load from disk — no rebuild needed
    vector_store = load_vector_store()

    while True:
        print()
        try:
            query = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # ── run the RAG pipeline ──────────────────────────────────────
        result = ask(vector_store, query)

        print("\n" + "─" * 55)
        print("[Answer]")
        print(result["answer"])

        # Show source passages (sources list is empty when fallback was triggered)
        if not result["is_fallback"] and result["sources"]:
            print("\n[Sources]")
            for src in result["sources"]:
                page_info = f"page {src['page']} · " if src["page"] != "" else ""
                print(
                    f"\n  > Passage {src['chunk_index']} "
                    f"({page_info}relevance {src['relevance_score']})"
                )
                print(f"    File:    {src['source_file']}")
                print(f"    Preview: {src['content_preview']}")

        print("─" * 55)


def main():
    """Parse the subcommand and dispatch to the appropriate function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py build   # build the vector store (run once, or after adding docs)")
        print("  python main.py ask     # start Q&A")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "build":
        cmd_build()
    elif command == "ask":
        cmd_ask()
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: build, ask")
        sys.exit(1)


if __name__ == "__main__":
    main()
