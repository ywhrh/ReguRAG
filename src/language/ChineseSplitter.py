import re

from langchain_core.documents import Document


class ChineseSplitter:
    """按"第X条"分割中文法规文档，保留原始 metadata。"""

    @staticmethod
    def split_by_article(text: str) -> list[str]:
        """把文本按条款边界切分，返回纯字符串列表。"""
        parts = re.split(r'(?=第[零一二三四五六七八九十百千]+条)', text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        """
        对 LangChain Document 列表按条款切分，返回新的 Document 列表。
        每个切片继承原文档的 metadata，并附加 article_index 字段。
        """
        result = []
        for doc in documents:
            articles = ChineseSplitter.split_by_article(doc.page_content)
            if not articles:
                result.append(doc)
                continue
            for i, article_text in enumerate(articles):
                result.append(Document(
                    page_content=article_text,
                    metadata={**doc.metadata, "article_index": i},
                ))
        return result
