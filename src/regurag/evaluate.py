import argparse
import csv
import datetime
import json
import os
import sys

import pandas as pd
from langchain_voyageai import VoyageAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import answer_relevancy, context_precision, context_recall, faithfulness

from regurag.config import (
    ANTHROPIC_API_KEY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RELEVANCE_THRESHOLD,
    TOP_K,
    VOYAGE_API_KEY,
)
from regurag.parser.english_only_llm import EnglishOnlyLLM
from regurag.qa_chain import ask
from regurag.vector_store import load_vector_store, retrieve_relevant_chunks

LOG_FILE = "optimization_log.csv"
RESULT_CACHE_FILE = "eval_cache.json"

LOG_COLUMNS = [
    "date",
    "chunk_size",
    "chunk_overlap",
    "relevance_threshold",
    "top_k",
    "total_questions",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "fallback_accuracy",
    "notes",
]


def load_test_set(csv_path: str) -> pd.DataFrame:
    """
    Load an evaluation CSV.

    Required columns: id, question, gold_answer, source_clause, type, notes.
    Supported type values: normal, hard, fallback.
    """
    if not os.path.exists(csv_path):
        print(f"Error: evaluation CSV not found: {csv_path}")
        print("Create an evaluation CSV using the schema described in README.md.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    counts = df["type"].value_counts().to_dict()
    print(f"Loaded {len(df)} evaluation row(s): {counts}")
    return df


def collect_rag_results(vector_store, df: pd.DataFrame) -> list:
    """
    Run the RAG pipeline for each evaluation row.

    The function performs retrieval once to capture full contexts for RAGAS, then
    calls ask() to reuse the same answer path as the app and CLI.
    """
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        question = str(row["question"])
        q_type = str(row.get("type", "normal"))
        gold = str(row.get("gold_answer", ""))

        print(f"\n[{idx + 1}/{total}] [{q_type}] {question[:55]}...")

        raw = retrieve_relevant_chunks(vector_store, question)
        max_score = max((s for _, s in raw), default=0.0)
        contexts = [doc.page_content for doc, s in raw if s >= RELEVANCE_THRESHOLD]
        qa = ask(vector_store, question)

        results.append({
            "question": question,
            "gold_answer": gold,
            "contexts": contexts,
            "answer": qa["answer"],
            "is_fallback": qa["is_fallback"],
            "max_score": max_score,
            "type": q_type,
        })

    return results


def setup_ragas_components():
    llm = EnglishOnlyLLM(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
    )

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-3",
    )

    return llm, embeddings


def run_ragas(results: list, llm, embeddings) -> dict:
    filtered = [r for r in results if r["type"] != "fallback"]
    if not filtered:
        print("No normal or hard questions found. Skipping RAGAS metrics.")
        return {}

    samples = []
    for r in filtered:
        contexts = r["contexts"] if r["contexts"] else ["No retrieved context passed the relevance threshold."]
        samples.append(SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=contexts,
            response=r["answer"],
            reference=r["gold_answer"],
        ))

    dataset = EvaluationDataset(samples=samples)

    print(f"\nRunning RAGAS evaluation for {len(samples)} question(s)...")

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    result_df = ragas_result.to_pandas()
    print("\nPer-question metric details:")
    print(result_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())

    scores: dict[str, float] = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if key in result_df.columns:
            col_mean = result_df[key].mean()
            if pd.notna(col_mean):
                scores[key] = round(float(col_mean), 4)

    return scores


def save_cache(results: list, path: str = RESULT_CACHE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Cached RAG results to {path}. Use --cache-answer to reuse them.")


def load_cache(path: str = RESULT_CACHE_FILE) -> list:
    if not os.path.exists(path):
        print(f"Error: cache file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} cached result(s).")
    return results


def eval_fallback(results: list):
    """Return fallback accuracy for rows marked type=fallback."""
    fb = [r for r in results if r["type"] == "fallback"]
    if not fb:
        return None, 0, 0
    correct = sum(1 for r in fb if r["is_fallback"])
    return correct / len(fb), correct, len(fb)


def _bar(val: float, width: int = 20) -> str:
    if val is None:
        return ""
    filled = round(val * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_results(scores: dict, fallback_acc, correct: int, fb_total: int):
    print("\n" + "=" * 62)
    print("  ReguRAG Evaluation Results")
    print("=" * 62)

    print("\n  RAGAS metrics for normal + hard rows\n")
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_precision": "Context precision",
        "context_recall": "Context recall",
    }
    for key, label in labels.items():
        val = scores.get(key)
        if val is not None:
            print(f"  {label:<18} {float(val):.4f}  {_bar(float(val))}")
        else:
            print(f"  {label:<18} N/A")

    print("\n  Fallback accuracy\n")
    if fallback_acc is not None:
        print(f"  {fallback_acc:.2%} ({correct}/{fb_total} fallback row(s))")
    else:
        print("  No fallback rows found.")


def append_log(scores: dict, fallback_acc, total_questions: int):
    row = {
        "date": datetime.date.today().isoformat(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "top_k": TOP_K,
        "total_questions": total_questions,
        "faithfulness": scores.get("faithfulness", ""),
        "answer_relevancy": scores.get("answer_relevancy", ""),
        "context_precision": scores.get("context_precision", ""),
        "context_recall": scores.get("context_recall", ""),
        "fallback_accuracy": round(fallback_acc, 4) if fallback_acc is not None else "",
        "notes": "",
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            existing_columns = f.readline().strip().split(",")
        if existing_columns != LOG_COLUMNS:
            backup = LOG_FILE.replace(".csv", f"_backup_{datetime.date.today().isoformat()}.csv")
            os.rename(LOG_FILE, backup)
            print(f"Log schema changed. Backed up old log to {backup}.")

    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nAppended evaluation row to {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(description="ReguRAG evaluation")
    parser.add_argument(
        "--csv",
        default="tests/eval/test_set_small.csv",
        help="Evaluation CSV path.",
    )
    parser.add_argument(
        "--cache-answer",
        action="store_true",
        help="Load cached RAG answers from eval_cache.json instead of regenerating them.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  ReguRAG Evaluation")
    print("=" * 62 + "\n")

    if args.cache_answer:
        results = load_cache()
        df = load_test_set(args.csv)
    else:
        df = load_test_set(args.csv)

        print("\nLoading vector store...")
        vector_store = load_vector_store()

        print("\nRunning RAG over the evaluation set...")
        results = collect_rag_results(vector_store, df)
        save_cache(results)

    fallback_acc, correct, fb_total = eval_fallback(results)
    ragas_llm, ragas_embeddings = setup_ragas_components()
    scores = run_ragas(results, ragas_llm, ragas_embeddings)

    print_results(scores, fallback_acc, correct, fb_total)
    append_log(scores, fallback_acc, len(df))


if __name__ == "__main__":
    main()
