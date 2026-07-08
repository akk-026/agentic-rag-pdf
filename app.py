import tempfile
from pathlib import Path

import streamlit as st

from src.agent import RAGAgent
from src.db import (
    create_chat,
    delete_chat,
    init_db,
    list_chats,
    message_count,
    rename_chat,
)
from src.docling_loader import load_pdf_documents
from src.vector_store import index_documents, reset_collection

st.set_page_config(
    page_title="Agentic RAG PDF Assistant",
    layout="wide",
)

init_db()

st.title("Agentic RAG PDF Assistant")
st.markdown(
    "Upload PDFs, ask questions, search the web, or use the calculator."
)

if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent()

if "selected_chat_id" not in st.session_state:
    chats = list_chats()
    st.session_state.selected_chat_id = chats[0]["id"] if chats else create_chat("New Chat")

with st.sidebar:
    st.header("Chats")

    if st.button("+ New Chat", use_container_width=True):
        new_chat_id = create_chat("New Chat")
        st.session_state.selected_chat_id = new_chat_id
        st.rerun()

    chats = list_chats()

    if chats:
        current_chat = next(
            (c for c in chats if c["id"] == st.session_state.selected_chat_id),
            chats[0],
        )

        selected_chat = st.selectbox(
            "Select chat",
            chats,
            index=chats.index(current_chat),
            format_func=lambda c: f"{c['title']}  ({c['message_count']} msgs)",
        )
        st.session_state.selected_chat_id = selected_chat["id"]

        if st.button("Delete Current Chat", use_container_width=True):
            delete_chat(st.session_state.selected_chat_id)
            chats_after = list_chats()
            if chats_after:
                st.session_state.selected_chat_id = chats_after[0]["id"]
            else:
                st.session_state.selected_chat_id = create_chat("New Chat")
            st.rerun()

    st.divider()

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF.")
        else:
            reset_collection()

            temp_paths = []
            for uploaded_file in uploaded_files:
                temp_dir = tempfile.mkdtemp()
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(str(file_path))

            with st.spinner(
    "Extracting text, chunking and indexing documents..."
):
                docs = load_pdf_documents(temp_paths)
                index_documents(docs)

            st.session_state.agent = RAGAgent()
            st.success(
                f"Indexed {len(docs)} chunks from {len(uploaded_files)} document(s)."
            )
    if st.button(
    "Clear Knowledge Base",
    use_container_width=True,):

        reset_collection()

        st.session_state.agent = RAGAgent()

        st.success("Knowledge base cleared.")

        st.rerun()
st.divider()

from src.db import get_messages  # local import to keep sidebar clean

messages = get_messages(st.session_state.selected_chat_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    if message_count(st.session_state.selected_chat_id) == 0:

        try:

            title = (
            st.session_state.agent.llm.invoke(
                f"Generate a very short chat title (max 5 words).\n\nQuestion:\n{question}"
            )
            .content.strip()
            .replace('"', "")
        )

        except Exception:

         title = " ".join(question.split()[:5])

        rename_chat(
        st.session_state.selected_chat_id,
        title,
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = st.write_stream(
                st.session_state.agent.stream_answer(
                    question,
                    chat_id=st.session_state.selected_chat_id,
                )
            )

            except Exception as e:

                st.error(str(e))

                answer = None
    if answer:
        st.rerun()