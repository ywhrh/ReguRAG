import re

from langchain_core.documents import Document


class ChineseSplitter:
    """Split Chinese regulatory text by article markers while preserving metadata."""

    @staticmethod
    def split_by_article(text: str) -> list[str]:
        """Split text at Chinese article boundaries."""
        parts = re.split(r'(?=第[零一二三四五六七八九十百千]+条)', text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        """
        Split LangChain documents by article.

        Each output document keeps the original metadata and receives an
        article_index field.
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
