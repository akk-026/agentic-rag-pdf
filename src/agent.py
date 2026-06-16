from typing import List, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.llm.factory import get_llm


class RAGState(TypedDict, total=False):
    original_query: str
    query: str
    chunks: List[RetrievedChunk]
    context: str
    answer: str
    retry_count: int
    needs_rewrite: bool


class RAGAgent:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = get_llm()
        self.max_retries = 1

        self.grade_prompt = ChatPromptTemplate.from_template(
            """
You are checking whether retrieved document chunks are enough to answer a question.

Question:
{question}

Retrieved context:
{context}

Reply with exactly one word:
SUFFICIENT
or
INSUFFICIENT
"""
        )

        self.rewrite_prompt = ChatPromptTemplate.from_template(
            """
Rewrite the user's question into a better search query for document retrieval.

Keep important entities, dates, document names, standards, and keywords.
Return only the rewritten query and nothing else.

Original question:
{question}
"""
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """
You are a careful enterprise document assistant.

Answer only using the provided context.
If the answer is not present in the context, say you could not find it.

Return a concise answer.
At the end, include a short Sources section using the source document name and page number.

Context:
{context}

Question:
{question}
"""
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("grade_documents", self._grade_documents_node)
        graph.add_node("rewrite_query", self._rewrite_query_node)
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_documents")

        graph.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "rewrite": "rewrite_query",
                "generate": "build_context",
            },
        )

        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _retrieve_node(self, state: RAGState):
        query = state.get("query") or state["original_query"]

        chunks = self.retriever.retrieve(
            query=query,
            mode="hybrid_rrf_rerank",
            top_k=5,
        )

        return {
            "query": query,
            "chunks": chunks,
        }

    def _grade_documents_node(self, state: RAGState):
        chunks = state.get("chunks", [])
        query = state.get("original_query") or state.get("query", "")

        if not chunks:
            return {"needs_rewrite": True}

        preview_parts = []
        for idx, chunk in enumerate(chunks[:3], start=1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "Unknown")
            preview_parts.append(
                f"[Chunk {idx}]\nSource: {source}\nPage: {page}\nContent:\n{chunk.content[:800]}"
            )

        context_preview = "\n\n".join(preview_parts)

        prompt_value = self.grade_prompt.invoke(
            {
                "question": query,
                "context": context_preview,
            }
        )

        decision = self.llm.generate(prompt_value.to_string()).strip().upper()

        return {
            "needs_rewrite": "INSUFFICIENT" in decision,
        }

    def _route_after_grading(self, state: RAGState) -> Literal["rewrite", "generate"]:
        retry_count = state.get("retry_count", 0)
        needs_rewrite = state.get("needs_rewrite", False)

        if needs_rewrite and retry_count < self.max_retries:
            return "rewrite"

        return "generate"

    def _rewrite_query_node(self, state: RAGState):
        original_query = state.get("original_query") or state.get("query", "")

        prompt_value = self.rewrite_prompt.invoke(
            {
                "question": original_query,
            }
        )

        rewritten_query = self.llm.generate(prompt_value.to_string()).strip()

        return {
            "query": rewritten_query or original_query,
            "retry_count": state.get("retry_count", 0) + 1,
            "needs_rewrite": False,
        }

    def _build_context_node(self, state: RAGState):
        parts = []

        for idx, chunk in enumerate(state.get("chunks", []), start=1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "Unknown")
            chunk_index = chunk.metadata.get("chunk_index", "Unknown")

            parts.append(
                f"""[Source {idx}]
Document: {source}
Page: {page}
Chunk: {chunk_index}
Content:
{chunk.content}
"""
            )

        context = "\n\n".join(parts).strip()

        if not context:
            context = "No relevant context retrieved."

        return {
            "context": context,
        }

    def _generate_node(self, state: RAGState):
        prompt_value = self.answer_prompt.invoke(
            {
                "context": state["context"],
                "question": state.get("original_query") or state.get("query", ""),
            }
        )

        answer = self.llm.generate(prompt_value.to_string())

        return {
            "answer": answer,
        }

    def answer(self, query: str) -> str:
        result = self.graph.invoke(
            {
                "original_query": query,
                "query": query,
                "retry_count": 0,
            }
        )
        return result["answer"]