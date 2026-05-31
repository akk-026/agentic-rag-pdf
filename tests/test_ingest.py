from pathlib import Path

from src.docling_loader import (
    load_pdf_documents,
    preview_documents
)

pdf_path = Path(
    "data/uploads/healthcare_india_hipaa.pdf"
)

docs = load_pdf_documents(str(pdf_path))

print(f"\nTotal chunks loaded: {len(docs)}\n")

preview_documents(docs, n=5)