from groq import Groq
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RELEVANCE_THRESHOLD,
    FALLBACK_MESSAGE,
)
from src.vector_store import retrieve_relevant_chunks


def build_prompt(query: str, relevant_chunks: list) -> str:
    """
    Construct the full prompt sent to Claude.

    Key design choices:
    1. XML tag separation: <regulations> wraps retrieved source text; <instructions>
       wraps behavioral constraints. This lets Claude clearly distinguish data from
       directives and prevents injected data from overriding instructions.
    2. Explicit fabrication ban: the instructions prohibit any answer not grounded
       in the provided passages.
    3. Mandatory citation: Claude must reference passage numbers so users can
       verify every claim against the original text.
    4. Language mirroring: Claude responds in the same language as the question.
    """
    # Build the regulations block, numbering each passage for easy citation
    regulations_text = ""
    for i, (doc, score) in enumerate(relevant_chunks, start=1):
        source_file = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        location = f"{source_file} (page {page})" if page != "" else source_file

        regulations_text += f"\n[Passage {i}] (Source: {location} | Relevance: {score:.3f})\n"
        regulations_text += doc.page_content.strip()
        regulations_text += "\n"

    prompt = f"""You are a professional financial regulatory Q&A assistant serving compliance officers and financial practitioners.

<regulations>
{regulations_text}
</regulations>

<instructions>
STRICT RULES — do not deviate:
1. Answer only based on the content inside <regulations>. Do not use any external knowledge, make inferences, or fabricate information.
2. If the provided passages do not contain a clear basis for an answer, explicitly state: "Based on the retrieved regulatory passages, no relevant provision was found." Do not draw any unsupported conclusion.
3. Every key claim must cite the relevant passage number (e.g., "According to [Passage 1]...") so the answer is fully traceable.
4. Reply in the same language as the user's question (Chinese question → Chinese answer; English question → English answer).
5. Use a professional and objective tone. When passages contain specific numbers, ratios, or deadlines, quote them exactly — do not paraphrase.
</instructions>

User question: {query}

Answer based on the regulatory passages above:"""

    return prompt


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
    }
    """
    # ── Step 1: vector retrieval ──────────────────────────────────────
    print("Retrieving relevant passages...")
    results = retrieve_relevant_chunks(vector_store, query)

    if not results:
        print("No results returned (vector store may be empty) — triggering fallback")
        return {"answer": FALLBACK_MESSAGE, "sources": [], "is_fallback": True, "max_score": 0.0}

    # ── Step 2: relevance threshold check (core anti-hallucination guard) ─
    # Use the highest score as the representative for this query
    max_score = max(score for _, score in results)
    print(f"Retrieved {len(results)} passage(s), top relevance: {max_score:.3f} (threshold: {RELEVANCE_THRESHOLD})")

    if max_score < RELEVANCE_THRESHOLD:
        # Relevance too low: the library has nothing matching this query.
        # Return the fixed fallback message without calling Claude at all.
        print("Relevance below threshold — returning fallback, LLM not called")
        return {"answer": FALLBACK_MESSAGE, "sources": [], "is_fallback": True, "max_score": max_score}

    # Keep only passages that clear the threshold
    relevant_chunks = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
    print(f"Passages passing threshold: {len(relevant_chunks)}/{len(results)}")

    # ── Step 3: build prompt ──────────────────────────────────────────
    prompt = build_prompt(query, relevant_chunks)

    # ── Step 4: call Groq to generate the answer ─────────────────────
    print("Calling Groq...")
    client = Groq(api_key=GROQ_API_KEY)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = completion.choices[0].message.content

    # ── Step 5: collect source metadata for display / logging ─────────
    sources = []
    for i, (doc, score) in enumerate(relevant_chunks, start=1):
        source_file = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        sources.append({
            "chunk_index": i,
            "source_file": source_file,
            "page": page,
            "relevance_score": round(score, 3),
            # Preview the first 200 characters so users can quickly verify
            "content_preview": (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            ),
        })

    return {
        "answer": answer,
        "sources": sources,
        "is_fallback": False,
        "max_score": max_score,
    }
