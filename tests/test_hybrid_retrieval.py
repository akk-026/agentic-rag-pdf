from src.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

query = "What was 3M's worldwide net sales in the second quarter of 2023?"

for mode in ["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank"]:
    print("\n" + "=" * 100)
    print(f"MODE: {mode}")

    results = retriever.retrieve(
        query=query,
        mode=mode,
        top_k=5,
        dense_k=20,
        bm25_k=20,
        rerank_pool_size=12,
    )

    for i, item in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank {i}")
        print(f"ID: {item.id}")
        print(f"Source: {item.metadata.get('source')}")
        print(f"Page: {item.metadata.get('page')}")
        print(item.content[:800])