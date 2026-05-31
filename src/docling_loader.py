import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_docling import DoclingLoader


def load_pdf_documents(file_paths: Union[str, List[str]]) -> List[Document]:
    """
    Load one or more PDFs using Docling and return LangChain Documents.

    DoclingLoader can accept a single local file path or an iterable of file paths.
    It returns chunk-oriented documents by default, which is good for RAG.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    normalized_paths = [str(Path(p)) for p in file_paths]

    loader = DoclingLoader(file_path=normalized_paths)
    docs = loader.load()

    return docs


def preview_documents(docs: List[Document], n: int = 3) -> None:
    """
    Small helper to inspect loaded chunks during development.
    """
    print(f"\nLoaded {len(docs)} document chunks\n")

    for i, doc in enumerate(docs[:n], start=1):
        print("=" * 80)
        print(f"Chunk {i}")
        print("Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:500])
        print()