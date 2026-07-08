from __future__ import annotations

from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent

from src.db import add_message, get_messages
from src.hybrid_retriever import HybridRetriever
from src.llm.factory import get_llm
from src.tools import calculator, web_search


class RAGAgent:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = get_llm()

        self.search_documents = self._build_search_documents_tool()
        self.tools = [self.search_documents, calculator, web_search]

        self.system_prompt = """
You are an intelligent enterprise document assistant.

You have access to three tools:

1. search_documents
- Searches the uploaded PDF documents.
- This should be your PRIMARY source of information.

2. web_search
- Searches the internet.
- Use ONLY if the uploaded documents do not contain the requested information or if the user explicitly asks for current/live information.

3. calculator
- Use ONLY for mathematical calculations or numerical expressions.

General Rules:

- Always answer in your own words.
- Never copy the raw output returned by any tool.
- Treat tool outputs as supporting evidence only.
- Combine information from multiple tool calls into one coherent answer.
- Keep answers concise unless the user requests detail.
- Never mention chunk numbers, retrieval IDs, reranker scores, or internal tool outputs.

Uploaded Documents:

- Assume uploaded PDFs are the primary knowledge source.
- Whenever a question could reasonably be answered using the uploaded PDFs, ALWAYS call search_documents FIRST.
- Do NOT use web_search until you have determined the uploaded documents do not contain the answer.
- If search_documents finds relevant information, answer using ONLY that information.
- Only fall back to web_search if the required information is missing from the uploaded PDFs.

Multiple Questions:

If the user asks multiple questions in one prompt:

1. Break the request into sub-questions.
2. Decide the best tool for each sub-question.
3. You may call tools multiple times.
4. Combine everything into one final response.
5. Do not stop after answering only one part.

Calculator:

Use the calculator tool whenever arithmetic or calculations are required.
Do not perform arithmetic mentally if the calculator can do it.

Answer Style:

- Answer naturally.
- Do not expose your reasoning.
- Do not print raw tool outputs.
- If information cannot be found, clearly state that.

At the end of every document-based answer, include:

Sources:
- Document Name (Page X)
- Document Name (Page Y)

Do NOT put citations inside sentences.

Never use formats like:
【1†source】
[1]
†
chunk numbers
retrieval ids
"""

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    def _build_search_documents_tool(self):
        retriever = self.retriever

        @tool("search_documents")
        def search_documents(query: str) -> str:
            """
            Search uploaded PDFs.
            Returns compact evidence for the LLM.
            """

            chunks = retriever.retrieve(
                query=query,
                mode="hybrid_rrf_rerank",
                top_k=3,
            )

            if not chunks:
                return "No relevant information found."

            evidence = []

            for chunk in chunks:

                source = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")

                text = " ".join(chunk.content.split())

                if len(text) > 450:
                    text = text[:450] + "..."

                evidence.append(
                    f"""
    Document: {source}
    Page: {page}

    {text}
    """
                )

            return "\n\n---\n\n".join(evidence)

        return search_documents

    def _load_history(self, chat_id: int) -> List[BaseMessage]:
        history: List[BaseMessage] = []

        for row in get_messages(chat_id,limit=20):
            role = row.get("role", "")
            content = row.get("content", "")

            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))

        return history

    def _build_inputs(self, query: str, chat_id: int) -> dict:
        messages = self._load_history(chat_id)
        messages.append(HumanMessage(content=query))
        return {"messages": messages}

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for piece in content:
                if isinstance(piece, str):
                    parts.append(piece)
                elif isinstance(piece, dict):
                    parts.append(piece.get("text", ""))
                else:
                    parts.append(getattr(piece, "text", "") or "")
            return "".join(parts)

        return str(content)

    def _extract_final_answer(self, result: dict) -> str:
        messages = result.get("messages", [])

        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = self._content_to_text(message.content).strip()
                if text:
                    return text

        return ""

    def _save_turn(self, chat_id: int, query: str, answer: str) -> None:
        if query:
            add_message(chat_id, "user", query)

        if answer:
            add_message(chat_id, "assistant", answer)

    def _run_agent(self, query: str, chat_id: int) -> dict:
        inputs = self._build_inputs(query, chat_id)
        return self.agent.invoke(
                inputs,
                config={
                "recursion_limit": 25
                },
                )

    def answer(self, query: str, chat_id: int) -> str:
        result = self._run_agent(query, chat_id)
        answer = self._extract_final_answer(result) or "I could not generate an answer."
        self._save_turn(chat_id, query, answer)
        return answer

    def stream_answer(self, query: str, chat_id: int):
        """
    Stream only the FINAL answer.
    """

        result = self._run_agent(query, chat_id)

        answer = (
            self._extract_final_answer(result)
            or "I could not generate an answer."
        )

        full_answer = ""

        for word in answer.split():
            token = word + " "
            full_answer += token
            yield token

        self._save_turn(chat_id, query, answer)