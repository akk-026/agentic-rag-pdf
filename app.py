import tempfile
from pathlib import Path

import streamlit as st

from src.agent import RAGAgent
from src.docling_loader import load_pdf_documents
from src.vector_store import index_documents, reset_collection


st.set_page_config(
    page_title="Agentic RAG PDF Assistant",
    layout="wide",
)

st.title("Agentic RAG PDF Assistant")


if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):

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

            with st.spinner("Processing documents..."):

                docs = load_pdf_documents(temp_paths)

                index_documents(docs)

            st.session_state.agent = RAGAgent()
            st.session_state.documents_loaded = True

            st.success(
                f"Indexed {len(docs)} chunks from {len(uploaded_files)} document(s)."
            )


st.divider()


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    if not st.session_state.documents_loaded:

        st.warning("Please upload and process documents first.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = st.session_state.agent.answer(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )