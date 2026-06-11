from src.docling_loader import load_pdf_documents
from src.vector_store import (
    reset_collection,
    index_documents,
    search
)

docs = load_pdf_documents([
    "data/uploads/healthcare_india_hipaa.pdf",
    "data/uploads/pdf.pdf"
])

reset_collection()

index_documents(docs)

result = search(
    "What was 3M net sales?"
)

for i, doc in enumerate(result["documents"][0], start=1):
    print("\n" + "=" * 80)
    print(f"Result {i}")
    
    print("\nMetadata:")
    print(result["metadatas"][0][i-1])

    print(doc[:1000])

    print(
        "\nDistance:",
        result["distances"][0][i - 1]
    )