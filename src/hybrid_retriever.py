import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import TOP_K_RESULTS
import src.vector_store as vector_store


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
    """

    def __init__(self, reranker_model: str = "BAAI/bge-reranker-base"):
        self.reranker = CrossEncoder(reranker_model)
        self.refresh()

    def refresh(self) -> None:
        """
        Reload all documents from Chroma and rebuild the BM25 index.
        Call this again after indexing new PDFs.
        """
        data = vector_store.collection.get(include=["documents", "metadatas"])

        self.ids = data.get("ids", []) or []
        self.documents = data.get("documents", []) or []
        self.metadatas = data.get("metadatas", []) or []

        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _dense_candidates(self, query: str, top_k: int) -> List[RetrievedChunk]:
        results = vector_store.search(query=query, top_k=top_k)

        dense_candidates: List[RetrievedChunk] = []

        for doc_id, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            dense_candidates.append(
                RetrievedChunk(
                    id=doc_id,
                    content=doc,
                    metadata=meta,
                    dense_distance=float(distance),
                )
            )

        return dense_candidates

    def _bm25_candidates(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        bm25_candidates: List[RetrievedChunk] = []

        for idx in ranked_indices:
            bm25_candidates.append(
                RetrievedChunk(
                    id=self.ids[idx],
                    content=self.documents[idx],
                    metadata=self.metadatas[idx],
                    bm25_score=float(scores[idx]),
                )
            )

        return bm25_candidates

    def _rrf_fuse(
        self,
        dense_candidates: List[RetrievedChunk],
        bm25_candidates: List[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.
        """
        fused = {}

        def add_score(candidate: RetrievedChunk, rank: int, source: str) -> None:
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

            fused[candidate.id].fusion_score = (fused[candidate.id].fusion_score or 0.0) + score

            # keep the best available metadata and scores
            if candidate.dense_distance is not None:
                fused[candidate.id].dense_distance = candidate.dense_distance
            if candidate.bm25_score is not None:
                fused[candidate.id].bm25_score = candidate.bm25_score

        for rank, candidate in enumerate(dense_candidates, start=1):
            add_score(candidate, rank, "dense")

        for rank, candidate in enumerate(bm25_candidates, start=1):
            add_score(candidate, rank, "bm25")

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

        rerank_pool = fused[:max(top_k, rerank_pool_size)]
        return self._rerank(query=query, candidates=rerank_pool, top_k=top_k)