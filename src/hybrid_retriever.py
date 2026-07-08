import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import TOP_K_RESULTS
import src.vector_store as vector_store
from src.retriever import build_bm25_retriever


_TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: dict
    dense_distance: Optional[float] = None
    bm25_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None


class HybridRetriever:
    """
    Dense search + BM25 + RRF fusion + BGE reranking.
    Uses LangChain vectorstore and BM25 retriever where practical.
    """

    def __init__(self, reranker_model: str = "BAAI/bge-reranker-base"):
        self.reranker = CrossEncoder(reranker_model)
        self.refresh()

    def refresh(self) -> None:
        """
        Reload all documents from Chroma and rebuild the BM25 retriever.
        """
        data = vector_store.collection.get(include=["documents", "metadatas"])

        raw_documents = data.get("documents", []) or []
        raw_metadatas = data.get("metadatas", []) or []
        raw_ids = data.get("ids", []) or []

        self.documents: List[Document] = []

        for i, text in enumerate(raw_documents):
            metadata = dict(raw_metadatas[i] or {})

            if i < len(raw_ids):
                metadata["id"] = raw_ids[i]
            else:
                source = metadata.get("source", "unknown")
                chunk_index = metadata.get("chunk_index", i)
                metadata["id"] = f"{source}_{chunk_index}"

            self.documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        self.ids = [doc.metadata["id"] for doc in self.documents]
        self.metadatas = [doc.metadata for doc in self.documents]

        self.bm25_retriever = (
            build_bm25_retriever(self.documents, k=20)
            if self.documents
            else None
        )

    def _dense_candidates(self, query: str, top_k: int) -> List[RetrievedChunk]:
        dense_candidates: List[RetrievedChunk] = []

        try:
            results = vector_store.vectorstore.similarity_search_with_score(
                query,
                k=top_k,
            )
        except Exception:
            docs = vector_store.vectorstore.similarity_search(query, k=top_k)
            results = [(doc, 0.0) for doc in docs]

        for doc, score in results:
            meta = dict(doc.metadata or {})
            doc_id = meta.get("id")
            if not doc_id:
                source = meta.get("source", "unknown")
                chunk_index = meta.get("chunk_index", 0)
                doc_id = f"{source}_{chunk_index}"

            dense_candidates.append(
                RetrievedChunk(
                    id=str(doc_id),
                    content=doc.page_content,
                    metadata=meta,
                    dense_distance=float(score) if score is not None else None,
                )
            )

        return dense_candidates

    def _bm25_candidates(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if self.bm25_retriever is None or not self.documents:
            return []

        self.bm25_retriever.k = top_k
        docs = self.bm25_retriever.invoke(query)

        bm25_candidates: List[RetrievedChunk] = []

        for rank, doc in enumerate(docs[:top_k], start=1):
            meta = dict(doc.metadata or {})
            doc_id = meta.get("id")
            if not doc_id:
                source = meta.get("source", "unknown")
                chunk_index = meta.get("chunk_index", 0)
                doc_id = f"{source}_{chunk_index}"

            bm25_candidates.append(
                RetrievedChunk(
                    id=str(doc_id),
                    content=doc.page_content,
                    metadata=meta,
                    bm25_score=1.0 / rank,
                )
            )

        return bm25_candidates

    def _rrf_fuse(
        self,
        dense_candidates: List[RetrievedChunk],
        bm25_candidates: List[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        fused = {}

        def add_score(candidate: RetrievedChunk, rank: int) -> None:
            score = 1.0 / (k + rank)

            if candidate.id not in fused:
                fused[candidate.id] = RetrievedChunk(
                    id=candidate.id,
                    content=candidate.content,
                    metadata=candidate.metadata,
                    dense_distance=candidate.dense_distance,
                    bm25_score=candidate.bm25_score,
                    fusion_score=0.0,
                )

            fused[candidate.id].fusion_score = (
                fused[candidate.id].fusion_score or 0.0
            ) + score

            if candidate.dense_distance is not None:
                fused[candidate.id].dense_distance = candidate.dense_distance
            if candidate.bm25_score is not None:
                fused[candidate.id].bm25_score = candidate.bm25_score

        for rank, candidate in enumerate(dense_candidates, start=1):
            add_score(candidate, rank)

        for rank, candidate in enumerate(bm25_candidates, start=1):
            add_score(candidate, rank)

        return sorted(
            fused.values(),
            key=lambda x: x.fusion_score or 0.0,
            reverse=True,
        )

    def _rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.content) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        return sorted(
            candidates,
            key=lambda x: x.rerank_score or 0.0,
            reverse=True,
        )[:top_k]

    def retrieve(
        self,
        query: str,
        mode: Literal["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank"] = "hybrid_rrf_rerank",
        top_k: int = TOP_K_RESULTS,
        dense_k: int = 20,
        bm25_k: int = 20,
        rerank_pool_size: int = 12,
    ) -> List[RetrievedChunk]:
        dense_candidates = self._dense_candidates(query=query, top_k=dense_k)
        bm25_candidates = self._bm25_candidates(query=query, top_k=bm25_k)

        if mode == "dense":
            return dense_candidates[:top_k]

        if mode == "bm25":
            return bm25_candidates[:top_k]

        fused = self._rrf_fuse(dense_candidates, bm25_candidates)

        if mode == "hybrid_rrf":
            return fused[:top_k]

        rerank_pool = fused[: max(top_k, rerank_pool_size)]
        return self._rerank(query=query, candidates=rerank_pool, top_k=top_k)