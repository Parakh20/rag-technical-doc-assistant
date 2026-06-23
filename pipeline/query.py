"""End-to-end query pipeline: query -> retrieve -> rerank -> generate."""

from __future__ import annotations

from core.generation import RAGGenerator, RAGResponse
from core.retrieval import DEFAULT_DENSE_K, DEFAULT_FINAL_K, RetrieverWithReranker
from core.vectorstore import ChromaStore


class QueryPipeline:
    def __init__(self, store: ChromaStore | None = None, generator: RAGGenerator | None = None):
        self.store = store or ChromaStore()
        self.retriever = RetrieverWithReranker(self.store)
        self.generator = generator or RAGGenerator()

    def ask(
        self,
        query: str,
        dense_k: int = DEFAULT_DENSE_K,
        final_k: int = DEFAULT_FINAL_K,
        use_reranker: bool = True,
        use_mmr: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> RAGResponse:
        if use_mmr:
            chunks = self.retriever.retrieve_mmr(query, dense_k=dense_k, final_k=final_k)
        else:
            chunks = self.retriever.retrieve(
                query, dense_k=dense_k, final_k=final_k, use_reranker=use_reranker
            )
        return self.generator.generate(query, chunks, conversation_history=conversation_history)


if __name__ == "__main__":
    pipeline = QueryPipeline()
    response = pipeline.ask("What is the maximum altitude for small UAS operations?")
    print(response.to_dict())
