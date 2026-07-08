"""Per-query observability: stage timings, token/cost estimate, and the
retrieved chunks' similarity distribution. Returned alongside RAGResponse
so the Streamlit UI (and anything else) can render it without recomputing
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Same rough per-1k-token estimate used in evaluation/metrics.py, kept here
# too rather than importing evaluation (core/ shouldn't depend on evaluation/).
ESTIMATED_COST_PER_1K_TOKENS_USD = 0.0003


@dataclass
class QueryMetrics:
    # retrieval_seconds covers the full RetrieverWithReranker.retrieve() call
    # (dense/hybrid search + optional query expansion/HyDE + cross-encoder
    # rerank) - these aren't separably timed without a deeper refactor of
    # retrieval.py, so they're reported as one stage rather than faking a
    # precision the current code doesn't have.
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_used: int
    num_chunks_retrieved: int
    similarity_scores: list[float] = field(default_factory=list)
    cache_hit: bool = False

    @property
    def estimated_cost_usd(self) -> float:
        return self.tokens_used / 1000 * ESTIMATED_COST_PER_1K_TOKENS_USD

    def to_dict(self) -> dict:
        return {
            "retrieval_seconds": self.retrieval_seconds,
            "generation_seconds": self.generation_seconds,
            "total_seconds": self.total_seconds,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "similarity_scores": self.similarity_scores,
            "cache_hit": self.cache_hit,
        }


if __name__ == "__main__":
    m = QueryMetrics(
        retrieval_seconds=0.42, generation_seconds=1.1, total_seconds=1.52,
        tokens_used=350, num_chunks_retrieved=5, similarity_scores=[0.8, 0.75, 0.6],
    )
    print(m.to_dict())
