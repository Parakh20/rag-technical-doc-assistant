"""Pydantic request/response models for the FastAPI query service.
QueryResponse mirrors core.generation.RAGResponse.to_dict() field-for-field
so the API contract stays in sync with the pipeline's actual return shape."""

from __future__ import annotations

from pydantic import BaseModel

from core.confidence import DEFAULT_REFUSAL_THRESHOLD
from core.retrieval import DEFAULT_DENSE_K, DEFAULT_FINAL_K


class QueryRequest(BaseModel):
    query: str
    dense_k: int = DEFAULT_DENSE_K
    final_k: int = DEFAULT_FINAL_K
    use_reranker: bool = True
    use_mmr: bool = False
    use_hybrid: bool = False
    use_query_expansion: bool = False
    use_hyde: bool = False
    use_correction: bool = False
    where: dict | None = None
    expand_to_parent: bool = False
    use_compression: bool = False
    compute_confidence_score: bool = True
    do_verify_citations: bool = False
    refusal_threshold: float = DEFAULT_REFUSAL_THRESHOLD
    conversation_history: list[dict] | None = None


class SourceRefResponse(BaseModel):
    filename: str
    page: int
    section: str
    relevance_score: float


class QueryMetricsResponse(BaseModel):
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_used: int
    estimated_cost_usd: float
    num_chunks_retrieved: int
    similarity_scores: list[float]
    cache_hit: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRefResponse]
    chunks_used: int
    tokens_used: int
    grounded: bool
    confidence: float | None
    refused: bool
    citations: list[dict] | None
    metrics: QueryMetricsResponse | None
