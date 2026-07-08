from typing import List

from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain_classic.retrievers import BM25Retriever


def build_bm25_retriever(
    documents: List[Document],
    k: int = 5,
):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever