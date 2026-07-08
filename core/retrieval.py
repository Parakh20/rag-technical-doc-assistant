"""Two-stage retrieval: dense search -> cross-encoder reranking, plus an
MMR (Maximal Marginal Relevance) alternative for diversity-aware selection.

Dense retrieval (embeddings) is fast but approximate - it compresses
semantics into a fixed-size vector. A cross-encoder scores the (query,
document) pair directly, which is slower but far more precise. Running
the cross-encoder only over the dense stage's top-20 candidates gives
near cross-encoder quality at a fraction of the cost of scoring the
whole corpus.
"""

from __future__ import annotations

from core.vectorstore import ChromaStore, SearchResult  # sets HF offline env vars first

from sentence_transformers import CrossEncoder

from core.bm25 import BM25Retriever
from core.fusion import reciprocal_rank_fusion
from core.query_transform import expand_query
from core.generation import RAGGenerator

DEFAULT_DENSE_K = 20
DEFAULT_FINAL_K = 5
DEFAULT_MMR_LAMBDA = 0.7
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrieverWithReranker:
    def __init__(
        self,
        store: ChromaStore,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        generator: RAGGenerator | None = None,
    ):
        self.store = store
        self.cross_encoder_model = cross_encoder_model
        self.generator = generator
        self._cross_encoder: CrossEncoder | None = None
        self._bm25: BM25Retriever | None = None

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def _get_bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.store)
        return self._bm25

    def _get_generator(self) -> RAGGenerator:
        if self.generator is None:
            self.generator = RAGGenerator()
        return self.generator

    def _dense_and_lexical(
        self, query: str, dense_k: int, where: dict | None, use_hybrid: bool
    ) -> list[SearchResult]:
        """One query's candidates: dense alone, or RRF-fused dense+BM25."""
        dense_results = self.store.search(query, k=dense_k, where=where)
        if not use_hybrid:
            return dense_results
        bm25_results = self._get_bm25().search(query, k=dense_k)
        if where:
            allowed_sources = {r.source for r in dense_results}
            bm25_results = [r for r in bm25_results if r.source in allowed_sources]
        return reciprocal_rank_fusion([dense_results, bm25_results])

    def retrieve(
        self,
        query: str,
        dense_k: int = DEFAULT_DENSE_K,
        final_k: int = DEFAULT_FINAL_K,
        use_reranker: bool = True,
        use_hybrid: bool = False,
        use_query_expansion: bool = False,
        where: dict | None = None,
        expand_to_parent: bool = False,
    ) -> list[SearchResult]:
        """Stage 1 dense (or hybrid dense+BM25, optionally multi-query) retrieval
        -> Stage 2 cross-encoder rerank -> top final_k -> optional parent-doc expansion."""
        queries = [query]
        if use_query_expansion:
            queries += expand_query(self._get_generator(), query)

        per_query_results = [
            self._dense_and_lexical(q, dense_k, where, use_hybrid) for q in queries
        ]
        candidates = (
            per_query_results[0]
            if len(per_query_results) == 1
            else reciprocal_rank_fusion(per_query_results)
        )
        if not candidates:
            return []
        if not use_reranker:
            top = candidates[:final_k]
            return self._expand_to_parent(top) if expand_to_parent else top

        cross_encoder = self._get_cross_encoder()
        pairs = [(query, c.text) for c in candidates]
        scores = cross_encoder.predict(pairs)
        reranked = [
            SearchResult(
                text=c.text, source=c.source, page=c.page,
                section=c.section, chunk_id=c.chunk_id, score=float(s),
                jurisdiction=c.jurisdiction, doc_type=c.doc_type,
                parent_id=c.parent_id, parent_text=c.parent_text,
            )
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        top = reranked[:final_k]
        return self._expand_to_parent(top) if expand_to_parent else top

    @staticmethod
    def _expand_to_parent(results: list[SearchResult]) -> list[SearchResult]:
        """Replace each result's (small) chunk text with its (larger) parent
        section/page text, deduplicating by parent_id so adjacent chunks
        from the same page don't produce repeated context - preserves the
        rank order of first appearance."""
        seen_parents: set[str] = set()
        expanded: list[SearchResult] = []
        for r in results:
            if not r.parent_id or r.parent_id in seen_parents:
                continue
            seen_parents.add(r.parent_id)
            expanded.append(
                SearchResult(
                    text=r.parent_text or r.text, source=r.source, page=r.page,
                    section=r.section, chunk_id=r.chunk_id, score=r.score,
                    jurisdiction=r.jurisdiction, doc_type=r.doc_type,
                    parent_id=r.parent_id, parent_text=r.parent_text,
                )
            )
        return expanded or results

    def retrieve_mmr(
        self,
        query: str,
        dense_k: int = DEFAULT_DENSE_K,
        final_k: int = DEFAULT_FINAL_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> list[SearchResult]:
        """Dense retrieval followed by MMR selection for query relevance +
        inter-result diversity. lambda_mult=1 is max relevance, 0 is max
        diversity."""
        candidates = self.store.search(query, k=dense_k)
        if not candidates:
            return []
        query_embedding = self.store.embedder.embed_query(query)
        doc_embeddings = self.store.embedder.embed_documents(
            [c.text for c in candidates], show_progress=False
        )
        selected_indices = self._mmr_select(
            query_embedding, doc_embeddings, final_k, lambda_mult
        )
        return [candidates[i] for i in selected_indices]

    @staticmethod
    def _mmr_select(query_embedding, doc_embeddings, k: int, lambda_mult: float) -> list[int]:
        relevance = doc_embeddings @ query_embedding  # embeddings are pre-normalized
        selected: list[int] = []
        remaining = list(range(len(doc_embeddings)))

        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                def mmr_score(i: int) -> float:
                    diversity = max(
                        float(doc_embeddings[i] @ doc_embeddings[j]) for j in selected
                    )
                    return lambda_mult * relevance[i] - (1 - lambda_mult) * diversity

                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader

    smoke_test_dir = Path(__file__).parent.parent / "vectorstore_data" / "_smoke_test_retrieval"
    shutil.rmtree(smoke_test_dir, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:30]
    chunks = chunk_pages(pages, strategy="recursive")
    print(f"Indexing {len(chunks)} chunks from {len(pages)} sample pages")

    store = ChromaStore(persist_dir=smoke_test_dir, collection_name="smoke_test_retrieval")
    store.add_documents(chunks)
    retriever = RetrieverWithReranker(store)

    query = "What are the requirements for visual line of sight operations?"

    print("\n-- Dense only (no reranker) --")
    for r in retriever.retrieve(query, final_k=3, use_reranker=False):
        print(f"  score={r.score:.3f} page={r.page} text={r.text[:80]!r}")

    print("\n-- Dense + cross-encoder rerank --")
    for r in retriever.retrieve(query, final_k=3, use_reranker=True):
        print(f"  score={r.score:.3f} page={r.page} text={r.text[:80]!r}")

    print("\n-- MMR (lambda=0.7) --")
    for r in retriever.retrieve_mmr(query, final_k=3):
        print(f"  score={r.score:.3f} page={r.page} text={r.text[:80]!r}")

    shutil.rmtree(smoke_test_dir, ignore_errors=True)
