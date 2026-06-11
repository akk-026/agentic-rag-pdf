from src.hybrid_retriever import HybridRetriever
from src.llm.factory import get_llm


class RAGAgent:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = get_llm()

    def _build_context(self, chunks) -> str:
        parts = []

        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "Unknown")

            parts.append(
                f"""[Source {idx}]
Document: {source}
Page: {page}
Content:
{chunk.content}
"""
            )

        return "\n\n".join(parts)

    def answer(self, query: str) -> str:
        chunks = self.retriever.retrieve(
            query=query,
            mode="hybrid_rrf_rerank",
            top_k=5,
        )

        context = self._build_context(chunks)

        prompt = f"""
You are a careful enterprise document assistant.

Answer the question only using the provided context.
If the answer is not in the context, say you could not find it.

Return a concise answer.
After the answer, include sources in this format:
- Document name (Page X)

Context:
{context}

Question:
{query}
"""

        return self.llm.generate(prompt)