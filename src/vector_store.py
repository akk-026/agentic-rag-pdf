import chromadb

from src.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME
)

from src.embeddings import (
    embed_texts,
    embed_query
)


# =====================================
# Create / Load Chroma Database
# =====================================

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)
def index_documents(docs):
    """
    Store LangChain documents in ChromaDB.
    """

    texts = [
        doc.page_content
        for doc in docs
    ]

    metadatas = [
        doc.metadata
        for doc in docs
    ]

    ids = [
        f"{doc.metadata['source']}_{i}"
        for i, doc in enumerate(docs)
    ]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Indexed {len(docs)} chunks")
def search(
    query: str,
    top_k: int = 5
):
    """
    Semantic search.
    """

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results
def reset_collection():
    """
    Delete all vectors.
    Useful during development.
    """

    global collection

    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME
    )

    print("Collection reset")