"""
ReguRAG 网页界面入口
启动方式：streamlit run app.py
访问地址：http://localhost:8501
"""
import sys
import os

import streamlit as st

# 把项目根目录加入 Python 路径，让 src/ 下的模块能正确 import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import load_vector_store
from src.qa_chain import ask


# ── 页面基本设置 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ReguRAG 金融法规问答", layout="centered")
st.title("ReguRAG 金融法规问答助手")
st.caption("基于检索增强生成（RAG），每条答案可溯源至法规原文，内置防幻觉机制。")


# ── 加载向量库（只在启动时加载一次，之后每次提问都复用缓存）────────────────────
@st.cache_resource(show_spinner="正在加载向量库，请稍候…")
def get_vector_store():
    """
    @st.cache_resource 让这个函数只在服务器首次启动时执行一次。
    之后用户每次提问都直接复用已加载的向量库，不会重复下载模型。
    """
    return load_vector_store()


# 尝试加载向量库；如果用户还没建库，给出清晰提示
try:
    vector_store = get_vector_store()
except FileNotFoundError:
    st.error(
        "未找到向量库（chroma_db/ 目录不存在）。\n\n"
        "请先在终端运行建库命令，再启动网页界面：\n\n"
        "```\npython main.py build\n```"
    )
    st.stop()  # 停止渲染页面后续内容


# ── 问答输入区 ────────────────────────────────────────────────────────────────
query = st.text_input(
    label="输入您的问题",
    placeholder="例如：商业银行核心一级资本充足率最低要求是多少？",
)

ask_clicked = st.button("提问", type="primary")


# ── 处理提问并展示结果 ────────────────────────────────────────────────────────
if ask_clicked and not query.strip():
    st.warning("请先输入问题再提交。")

elif ask_clicked and query.strip():
    with st.spinner("检索中，请稍候…"):
        result = ask(vector_store, query.strip())

    st.divider()

    # ── 答案区 ────────────────────────────────────────────────────────────────
    st.subheader("答案")
    st.markdown(result["answer"])

    # 最高相关度分数（方便调试，了解检索质量）
    st.caption(f"最高相关度分数：{result['max_score']:.3f}（防幻觉阈值：0.3）")

    if result["is_fallback"]:
        # 触发兜底逻辑：相关度不足，LLM 未被调用
        st.warning("相关度低于阈值，未调用 LLM，以上为兜底提示。")

    else:
        # ── 引用来源区（本项目核心特性）────────────────────────────────────────
        st.subheader("引用来源")
        st.caption("以下是本次回答所依据的法规原文片段，可自行核对：")

        for src in result["sources"]:
            # 有页码时显示页码，纯 txt 文件没有页码则跳过
            page_info = f" · 第 {src['page']} 页" if src["page"] != "" else ""
            label = (
                f"片段 {src['chunk_index']}"
                f"｜相关度 {src['relevance_score']}"
                f"{page_info}"
            )

            with st.expander(label):
                st.text(f"来源文件：{src['source_file']}")
                st.markdown(f"> {src['content_preview']}")
