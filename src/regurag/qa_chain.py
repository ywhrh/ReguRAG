import anthropic

from regurag.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    RELEVANCE_THRESHOLD,
    FALLBACK_MESSAGE,
)
from regurag.vector_store import retrieve_relevant_chunks

# Static system instructions are eligible for Anthropic prompt caching.
_SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": (
            "You are a professional financial regulatory Q&A assistant "
            "serving compliance officers and financial practitioners.\n\n"
            "<instructions>\n"
            "STRICT RULES — do not deviate:\n"
            "1. Answer only based on the content inside <regulations>. "
            "Do not use any external knowledge, make inferences, or fabricate information.\n"
            "2. If the provided passages do not contain a clear basis for an answer, "
            'explicitly state: "Based on the retrieved regulatory passages, no relevant provision was found." '
            "Do not draw any unsupported conclusion.\n"
            "3. Every key claim must cite the relevant passage number "
            '(e.g., "According to [Passage 1]...") so the answer is fully traceable.\n'
            "4. Reply in the same parser as the user's question "
            "(Chinese question → Chinese answer; English question → English answer).\n"
            "5. Use a professional and objective tone. When passages contain specific numbers, "
            "ratios, or deadlines, quote them exactly — do not paraphrase.\n"
            "</instructions>"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


def _build_user_message(query: str, relevant_chunks: list) -> str:
    """Build the dynamic prompt content for the retrieved passages and user question."""
    regulations_text = ""
    for i, (doc, score) in enumerate(relevant_chunks, start=1):
        source_file = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        location = f"{source_file} (page {page})" if page != "" else source_file
        regulations_text += f"\n[Passage {i}] (Source: {location} | Relevance: {score:.3f})\n"
        regulations_text += doc.page_content.strip()
        regulations_text += "\n"

    return (
        f"<regulations>\n{regulations_text}</regulations>\n\n"
        f"User question: {query}\n\n"
        "Answer based on the regulatory passages above:"
    )


def _format_sources(relevant_chunks: list) -> list:
    """Collect source metadata for display / logging."""
    sources = []
    for i, (doc, score) in enumerate(relevant_chunks, start=1):
        source_file = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        sources.append({
            "chunk_index": i,
            "source_file": source_file,
            "page": page,
            "relevance_score": round(score, 3),
            "content_preview": (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            ),
        })
    return sources


def ask(vector_store, query: str) -> dict:
    """
    Run the full RAG pipeline and return the answer with source metadata.

    Pipeline:
      retrieve → threshold check (anti-hallucination) → build prompt → call Claude → collect sources

    Return shape:
    {
        "answer":      str,   # final answer text
        "sources":     list,  # source passage list (empty when is_fallback is True)
        "is_fallback": bool,  # True means the LLM was never called
        "max_score":   float, # highest relevance score (useful for debugging)
        "api_error":   str,   # optional LLM/API error detail
    }
    """
    print("Retrieving relevant passages...")
    results = retrieve_relevant_chunks(vector_store, query)

    if not results:
        print("No results returned (vector store may be empty) — triggering fallback")
        return {"answer": FALLBACK_MESSAGE, "sources": [], "is_fallback": True, "max_score": 0.0, "api_error": ""}

    max_score = max(score for _, score in results)
    print(f"Retrieved {len(results)} passage(s), top relevance: {max_score:.3f} (threshold: {RELEVANCE_THRESHOLD})")

    if max_score < RELEVANCE_THRESHOLD:
        print("Relevance below threshold — returning fallback, LLM not called")
        return {"answer": FALLBACK_MESSAGE, "sources": [], "is_fallback": True, "max_score": max_score, "api_error": ""}

    relevant_chunks = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
    print(f"Passages passing threshold: {len(relevant_chunks)}/{len(results)}")
    sources = _format_sources(relevant_chunks)

    print("Calling Claude...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, relevant_chunks)}],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
    except anthropic.AnthropicError as exc:
        error_text = str(exc)
        print(f"Claude API call failed: {error_text}")
        return {
            "answer": (
                "Relevant regulatory passages were retrieved, but the answer-generation "
                "API call failed. Please check the API key, quota, model name, and network access."
            ),
            "sources": sources,
            "is_fallback": False,
            "max_score": max_score,
            "api_error": error_text,
        }

    answer = message.content[0].text

    return {
        "answer": answer,
        "sources": sources,
        "is_fallback": False,
        "max_score": max_score,
        "api_error": "",
    }
