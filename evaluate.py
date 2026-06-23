import argparse
import csv
import datetime
import os
import sys

import pandas as pd
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall

from src.language.EnglishOnlyLLM import EnglishOnlyLLM

# 把项目根目录加入 Python 路径，让 src/ 模块能正确 import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ANTHROPIC_API_KEY, RELEVANCE_THRESHOLD, VOYAGE_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from src.vector_store import load_vector_store, retrieve_relevant_chunks
from src.qa_chain import ask
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_voyageai import VoyageAIEmbeddings

LOG_FILE = "optimization_log.csv"

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


# ── Step 1：读取测试集 ─────────────────────────────────────────────────────────

def load_test_set(csv_path: str) -> pd.DataFrame:
    """
    读取测试集 CSV，返回 DataFrame。
    必须包含这几列：id, question, gold_answer, source_clause, type, notes
    type 的取值：normal（普通问题）/ hard（难题）/ fallback（应触发兜底的）
    """
    if not os.path.exists(csv_path):
        print(f"错误：找不到测试集 {csv_path}")
        print("请先准备 test_set.csv，格式参考 README 中的说明。")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    counts = df["type"].value_counts().to_dict()
    print(f"已加载测试集：{len(df)} 条  |  类型分布：{counts}")
    return df


# ── Step 2：对每道题跑 RAG 流程，收集数据 ────────────────────────────────────

def collect_rag_results(vector_store, df: pd.DataFrame) -> list:
    """
    对测试集里的每道题跑一次完整 RAG 流程，收集以下内容：
      - question       ：用户问题
      - gold_answer    ：测试集里的标准答案（供 RAGAS context_recall 使用）
      - contexts       ：检索到并通过阈值的法规片段全文列表（RAGAS 需要全文）
      - answer         ：系统生成的答案（兜底时为兜底消息）
      - is_fallback    ：是否触发了兜底（True = LLM 未被调用）
      - max_score      ：最高相关度分数（用于调试）
      - type           ：问题类型（normal / hard / fallback）

    注意：每题会调用两次检索。
      第一次（在这个函数里）：为了拿到完整的 contexts 文本，供 RAGAS 使用。
        ask() 的 sources 里只有 200 字预览，RAGAS 需要完整原文。
      第二次（在 ask() 内部）：获取最终答案。
    这是为了不改动现有 ask() 函数的逻辑，评估脚本可以接受这点额外开销。
    """
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        question = str(row["question"])
        q_type = str(row.get("type", "normal"))
        gold = str(row.get("gold_answer", ""))

        print(f"\n[{idx + 1}/{total}] [{q_type}] {question[:55]}...")

        # 检索：拿到所有候选片段及相关度分数
        raw = retrieve_relevant_chunks(vector_store, question)
        max_score = max((s for _, s in raw), default=0.0)

        # 只保留通过阈值的片段全文（和 ask() 内部的过滤逻辑一致）
        contexts = [doc.page_content for doc, s in raw if s >= RELEVANCE_THRESHOLD]

        # 调用现有问答逻辑，拿到最终答案
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
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
    )

    # Embedding 用 Voyage（Anthropic 官方合作）
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-3",  # 通用场景
        # model="voyage-finance-2" # 金融场景专用
    )

    # return ragas_llm, ragas_embeddings
    return llm, embeddings


def run_ragas(results: list, llm, embeddings) -> dict:
    # 过滤掉 fallback 类
    filtered = [r for r in results if r["type"] != "fallback"]
    if not filtered:
        print("没有 normal / hard 类问题，跳过 RAGAS 指标计算。")
        return {}

    # 构建 RAGAS 所需的数据集格式
    samples = []
    for r in filtered:
        # 如果没有任何片段通过阈值（比如 normal 题也被兜底），用占位文本
        # 这道题的指标分数会很低，属于预期行为，说明检索有问题
        contexts = r["contexts"] if r["contexts"] else ["（无相关片段通过阈值）"]
        samples.append(SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=contexts,  # 检索到的法规片段全文列表
            response=r["answer"],  # 系统生成的答案
            reference=r["gold_answer"],  # 标准答案（context_recall 需要）
        ))

    dataset = EvaluationDataset(samples=samples)

    print(f"\n运行 RAGAS 评估（{len(samples)} 道题，每题多次 API 调用，预计 1-5 分钟）…")

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    # EvaluationResult 不支持 .get()，转成 DataFrame 后对每列取均值
    result_df = ragas_result.to_pandas()
    print("\n[DEBUG] 每题分数明细：")
    print(result_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())
    scores: dict[str, float] = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if key in result_df.columns:
            col_mean = result_df[key].mean()
            if pd.notna(col_mean):
                scores[key] = round(float(col_mean), 4)

    return scores


# ── Step 5：单独评估 fallback 类 ──────────────────────────────────────────────

def eval_fallback(results: list):
    """
    统计 fallback 类问题的兜底正确率。

    正确行为 = 系统触发了兜底（is_fallback=True）。
      → 说明系统识别出检索相关度不够，没有把问题发给 LLM，避免了幻觉。

    错误行为 = 系统没触发兜底，把不相关的问题发给了 LLM。
      → 这种情况下 LLM 可能编造不存在的法规条款，危害大。

    返回：(正确率, 正确条数, 总条数)
    """
    fb = [r for r in results if r["type"] == "fallback"]
    if not fb:
        return None, 0, 0
    correct = sum(1 for r in fb if r["is_fallback"])
    return correct / len(fb), correct, len(fb)


# ── Step 6：打印结果 ──────────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    """把 0-1 的分数画成一个小进度条，方便一眼看出高低。"""
    if val is None:
        return ""
    filled = round(val * width)
    return "[" + "█" * filled + "░" * (20 - filled) + "]"


def print_results(scores: dict, fallback_acc, correct: int, fb_total: int, results: list):
    """在终端打印评估结果汇总。"""
    print("\n" + "=" * 62)
    print("  ReguRAG 评估结果")
    print("=" * 62)

    # ── RAGAS 指标 ──
    print("\n  ▌ RAGAS 指标（normal + hard 类，范围 0-1，越高越好）\n")
    labels = {
        "faithfulness": "忠实度（防幻觉）",
        "answer_relevancy": "答案相关性      ",
        "context_precision": "检索精准度      ",
        "context_recall": "检索召回率      ",
    }
    for key, label in labels.items():
        val = scores.get(key)
        if val is not None:
            print(f"  {label}  {float(val):.4f}  {_bar(float(val))}")
        else:
            print(f"  {label}  N/A")

    # ── fallback 正确率 ──
    print("\n  ▌ 兜底正确率（fallback 类）\n")
    if fallback_acc is not None:
        print(f"  {fallback_acc:.2%}  （{correct}/{fb_total} 条正确触发兜底）")
    else:
        print("  测试集中无 fallback 类，跳过。")

    # ── 逐题明细（调试用）──
    # print("\n  ▌ 逐题明细\n")
    # print(f"  {'类型':<8}  {'相关度':>6}  {'兜底':^4}  问题")
    # print("  " + "-" * 56)
    # for r in results:
    #     marker = "✓ 是" if r["is_fallback"] else "✗ 否"
    #     print(f"  {r['type']:<8}  {r['max_score']:>6.3f}  {marker:^4}  {r['question'][:35]}")
    #
    # print("\n" + "=" * 62)


# ── Step 7：追加调参日志 ──────────────────────────────────────────────────────

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

    # 如果日志已存在但列定义与代码不一致，备份旧文件并新建
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            existing_columns = f.readline().strip().split(",")
        if existing_columns != LOG_COLUMNS:
            backup = LOG_FILE.replace(".csv", f"_backup_{datetime.date.today().isoformat()}.csv")
            os.rename(LOG_FILE, backup)
            print(f"列定义已变更，旧日志已备份为 {backup}，创建新日志。")

    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n结果已追加到 {LOG_FILE}")
    print("建议打开文件，手动填写 change_made 和 notes 两列，方便日后对比。")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ReguRAG 自动评估")
    parser.add_argument(
        "--csv", default="src/test/test_set.csv",
        help="测试集路径（默认 ./test_set.csv）"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  ReguRAG 自动评估")
    print("=" * 62 + "\n")

    # 1. 加载测试集
    df = load_test_set(args.csv)

    # 2. 加载向量库（需要先运行 python main.py build）
    print("\n加载向量库…")
    vector_store = load_vector_store()

    # 3. 对每道题跑 RAG 流程，收集数据
    print("\n对测试集每道题运行 RAG 流程，收集结果…")
    results = collect_rag_results(vector_store, df)

    # 4. 单独统计 fallback 类的兜底正确率（不依赖 RAGAS）
    fallback_acc, correct, fb_total = eval_fallback(results)

    # 5. 配置 RAGAS 组件
    ragas_llm, ragas_embeddings = setup_ragas_components()

    # 6. 运行 RAGAS 评估（只针对 normal / hard 类）
    scores = run_ragas(results, ragas_llm, ragas_embeddings)

    print_results(scores, fallback_acc, correct, fb_total, results)

    # 8. 追加到调参日志
    append_log(scores, fallback_acc, len(df))


if __name__ == "__main__":
    main()
