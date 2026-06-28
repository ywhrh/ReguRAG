"""Streamlit web UI for ReguRAG."""

import streamlit as st

from regurag.config import ANTHROPIC_API_KEY, DATA_DIR, RELEVANCE_THRESHOLD
from regurag.document_loader import get_supported_document_files
from regurag.qa_chain import ask
from regurag.vector_store import load_vector_store


st.set_page_config(page_title="ReguRAG Financial Regulatory Q&A", layout="centered")
st.title("ReguRAG Financial Regulatory Q&A")
st.caption("Retrieval-augmented answers with source citations and a relevance guardrail.")


source_files = get_supported_document_files(DATA_DIR)

with st.sidebar:
    st.header("Source Files")
    st.caption(f"Loaded from `{DATA_DIR}/`")

    if source_files:
        st.success(f"{len(source_files)} file(s) available")
        for filename in source_files:
            st.text(filename)
    else:
        st.warning("No .txt or .pdf files found.")
        st.markdown(
            "Add regulatory documents to `data/`, then rebuild the index:\n\n"
            "```bash\n"
            "PYTHONPATH=src python -m regurag.main build\n"
            "```"
        )


if not source_files:
    st.error(
        "No regulatory source files were found.\n\n"
        "Add `.txt` or `.pdf` files to the `data/` directory, then rebuild the Chroma index."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    """Load the vector store once per Streamlit server process."""
    return load_vector_store()


try:
    vector_store = get_vector_store()
except FileNotFoundError:
    st.error(
        "Vector store not found: `chroma_db/` does not exist.\n\n"
        "Build the index from the source files shown in the sidebar before asking questions:\n\n"
        "```\nPYTHONPATH=src python -m regurag.main build\n```"
    )
    st.stop()


with st.form("question_form"):
    query = st.text_input(
        label="Question",
        placeholder="Example: What is the minimum Common Equity Tier 1 capital requirement?",
    )
    ask_clicked = st.form_submit_button("Ask", type="primary")


if ask_clicked and not query.strip():
    st.warning("Enter a question before submitting.")

elif ask_clicked and query.strip():
    if not ANTHROPIC_API_KEY:
        st.error(
            "`ANTHROPIC_API_KEY` is not set.\n\n"
            "Add it to your local `.env` file or pass it to Docker:\n\n"
            "```\n"
            "docker run --env-file .env -p 8501:8501 regurag\n"
            "```"
        )
        st.stop()

    with st.spinner("Retrieving relevant passages..."):
        result = ask(vector_store, query.strip())

    st.divider()

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.caption(f"Top relevance score: {result['max_score']:.3f} (threshold: {RELEVANCE_THRESHOLD})")

    if result["is_fallback"]:
        st.warning("Relevance was below the threshold, so the LLM was not called.")

    else:
        if result.get("api_error"):
            st.error("The LLM call failed. Retrieved passages are still shown for debugging.")
            with st.expander("Error details"):
                st.code(result["api_error"])

        st.subheader("Sources")
        st.caption("Retrieved regulatory passages used for this answer.")

        for src in result["sources"]:
            page_info = f" · page {src['page']}" if src["page"] != "" else ""
            label = (
                f"Passage {src['chunk_index']}"
                f" | relevance {src['relevance_score']}"
                f"{page_info}"
            )

            with st.expander(label):
                st.text(f"Source file: {src['source_file']}")
                st.markdown(f"> {src['content_preview']}")
