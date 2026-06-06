from src.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    reranker_model="BAAI/bge-reranker-base"
)

query = "What is ISO 27789?"

modes = [
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
]

for mode in modes:
    print("\n" + "=" * 100)
    print(f"MODE: {mode}")

    results = retriever.retrieve(
        query=query,
        mode=mode,
        top_k=5,
        dense_k=10,
        bm25_k=10,
        rerank_pool_size=8,
    )

    for i, item in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank {i}")
        print(f"ID: {item.id}")
        print(f"Distance: {item.dense_distance}")
        print(f"BM25: {item.bm25_score}")
        print(f"Fusion: {item.fusion_score}")
        print(f"Rerank: {item.rerank_score}")
        print(f"Source: {item.metadata.get('source')}")
        print(item.content[:800])