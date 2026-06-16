import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import CHROMA_DB_DIR, COLLECTION_NAME
from src.embeddings import embed_query, embed_texts


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        return embed_query(text)


client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_embeddings = LocalEmbeddings()

# LangChain Chroma wrapper on top of the same persistent Chroma collection
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    client=client,
    embedding_function=_embeddings,
)

# Raw collection still kept for compatibility / debugging / tests
collection = vectorstore._collection


def index_documents(docs: list[Document]) -> None:
    """
    Store LangChain Documents in ChromaDB.
    """
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [
        f"{doc.metadata['source']}_{doc.metadata['chunk_index']}"
        for doc in docs
    ]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(docs)} chunks")


def search(query: str, top_k: int = 5):
    """
    Semantic search using the raw Chroma collection.
    Kept for backwards compatibility with your tests.
    """
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results


def reset_collection() -> None:
    """
    Delete all vectors and recreate the collection.
    Useful during development.
    """
    global collection, vectorstore

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        client=client,
        embedding_function=_embeddings,
    )
    collection = vectorstore._collection

    print("Collection reset")