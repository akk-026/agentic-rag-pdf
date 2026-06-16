from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.llm.factory import get_llm


class RAGState(TypedDict, total=False):
    query: str
    chunks: List[RetrievedChunk]
    context: str
    answer: str


class RAGAgent:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_template(
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
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _retrieve_node(self, state: RAGState):
        chunks = self.retriever.retrieve(
            query=state["query"],
            mode="hybrid_rrf_rerank",
            top_k=5,
        )
        return {"chunks": chunks}

    def _build_context_node(self, state: RAGState):
        parts = []

        for idx, chunk in enumerate(state.get("chunks", []), start=1):
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

        return {"context": "\n\n".join(parts)}

    def _generate_node(self, state: RAGState):
        prompt_value = self.prompt.invoke(
            {
                "context": state["context"],
                "question": state["query"],
            }
        )

        answer = self.llm.generate(prompt_value.to_string())
        return {"answer": answer}

    def answer(self, query: str) -> str:
        result = self.graph.invoke({"query": query})
        return result["answer"]