"""End-to-end query pipeline: query -> retrieve -> rerank -> generate."""

from __future__ import annotations

import dataclasses

from core.citation_check import verify_citations
from core.compression import compress_chunks
from core.confidence import DEFAULT_REFUSAL_THRESHOLD, compute_confidence, should_refuse
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
        use_hybrid: bool = False,
        use_query_expansion: bool = False,
        use_hyde: bool = False,
        use_correction: bool = False,
        where: dict | None = None,
        expand_to_parent: bool = False,
        use_compression: bool = False,
        compute_confidence_score: bool = True,
        do_verify_citations: bool = False,
        refusal_threshold: float = DEFAULT_REFUSAL_THRESHOLD,
        conversation_history: list[dict] | None = None,
    ) -> RAGResponse:
        if use_mmr:
            chunks = self.retriever.retrieve_mmr(query, dense_k=dense_k, final_k=final_k)
        elif use_correction:
            chunks, _iterations = self.retriever.retrieve_with_correction(
                query, dense_k=dense_k, final_k=final_k, use_hybrid=use_hybrid,
                where=where, expand_to_parent=expand_to_parent,
            )
        else:
            chunks = self.retriever.retrieve(
                query, dense_k=dense_k, final_k=final_k, use_reranker=use_reranker,
                use_hybrid=use_hybrid, use_query_expansion=use_query_expansion,
                use_hyde=use_hyde, where=where, expand_to_parent=expand_to_parent,
            )
        if use_compression:
            chunks = compress_chunks(query, chunks, self.retriever.store.embedder)

        response = self.generator.generate(query, chunks, conversation_history=conversation_history)

        confidence = None
        refused = False
        if compute_confidence_score:
            confidence = compute_confidence(chunks[0] if chunks else None)
            refused = should_refuse(confidence, refusal_threshold)

        citations = None
        if do_verify_citations and chunks:
            verifications = verify_citations(
                response.answer, chunks, self.retriever._get_cross_encoder()
            )
            citations = [v.to_dict() for v in verifications]

        return dataclasses.replace(
            response, confidence=confidence, refused=refused, citations=citations
        )


if __name__ == "__main__":
    pipeline = QueryPipeline()
    response = pipeline.ask("What is the maximum altitude for small UAS operations?")
    print(response.to_dict())
