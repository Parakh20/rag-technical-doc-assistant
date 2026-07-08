# Advanced RAG Upgrade — Design

## Goal

Take the existing dense + cross-encoder-reranked RAG pipeline
(`core/retrieval.py`, `core/generation.py`, `pipeline/query.py`,
`evaluation/metrics.py`) from "classical RAG" to a system that demonstrates
hybrid retrieval, context shaping, self-correction, and production-grade
evaluation/observability — while keeping every added LLM-call feature
toggleable so cost/latency stays under the operator's control on a
rate-limited free-tier Gemini key.

Out of scope (explicitly rejected during brainstorming): GraphRAG, ColBERT
late-interaction, LightRAG, LongRAG, Fusion-in-Decoder, Memory RAG. These
either require an architecture change bigger than a bolt-on (ColBERT,
Fusion-in-Decoder) or overlap with items already in scope (LongRAG ≈
parent-doc retrieval, Memory RAG ≈ existing conversation history support).

## Architecture (end state)

```
Query
  │
  ├─ [optional] HyDE: LLM writes hypothetical answer → embed that
  │
  ├─ [optional] Query expansion: LLM reformulations (N variants)
  │
  ▼
┌─────────────────────────────────────────────┐
│ Hybrid retrieval (per query variant, gathered concurrently) │
│   BM25Retriever (rank_bm25, in-memory)       │
│   VectorStore.search() (Chroma/FAISS/Qdrant) │
│        with optional metadata `where` filter │
└─────────────────────────────────────────────┘
  │
  ▼
Reciprocal Rank Fusion (merge BM25 + dense + query-variant lists)
  │
  ▼
Cross-encoder rerank (existing) → top-k
  │
  ├─ [optional] CRAG/agentic grading: are top-k good enough?
  │     no → rewrite query & re-retrieve, or issue sub-query (bounded loop)
  │
  ├─ [optional] Parent-document expansion: child chunk → parent section text
  │
  ├─ [optional] Context compression: extractive sentence selection
  │
  ▼
Generate (existing Gemini + fallback), streamed over SSE from FastAPI
  │
  ├─ Confidence estimation (dense + cross-encoder + groundedness) → refuse if low
  ├─ [optional] Citation verification (sentence-level entailment vs cited chunk)
  │
  ▼
Cited, confidence-scored, (optionally) verified answer + per-query metrics
```

Indexing side gains:
- Metadata tagging (`jurisdiction`, `doc_type`) at ingestion.
- Parent/child chunk pairs stored together.
- Optional RAPTOR tree built as a separate offline step, persisted next to
  the vector store.

Retrieval cache (in-memory LRU keyed on normalized query + active
flags/filters) sits in front of the whole pipeline in the FastAPI service.

## Components

| Component | File | Responsibility |
|---|---|---|
| `BM25Retriever` | `core/bm25.py` (new) | In-memory BM25 index over the same chunk corpus; same `SearchResult` output shape as `ChromaStore.search`. |
| RRF merge | `core/fusion.py` (new) | Pure function: `list[list[SearchResult]] -> list[SearchResult]`, reciprocal rank fusion. |
| Query expansion / HyDE | `core/query_transform.py` (new) | LLM-backed: `expand_query(query) -> list[str]`, `hyde_embed(query) -> vector`. Each gated by a flag; reuses `RAGGenerator`'s client. |
| `VectorStore` Protocol | `core/vectorstore.py` (extend) | `add_documents`, `search`, `get_stats` as a `Protocol`; `ChromaStore` annotated as conforming. |
| `FaissStore`, `QdrantStore` | `core/vectorstore_faiss.py`, `core/vectorstore_qdrant.py` (new) | Alternate `VectorStore` implementations, same interface, selected via config/env. |
| Parent-doc retrieval | `core/chunking.py` (extend), `core/retrieval.py` (extend) | Chunking emits child chunks with a `parent_id`/`parent_text` field; retrieval has an `expand_to_parent` flag. |
| Context compression | `core/compression.py` (new) | Extractive: sentence-split each chunk, score by cosine sim to query embedding, keep top sentences under a token budget. |
| CRAG/agentic loop | `core/retrieval.py` (extend) | `retrieve_with_correction()`: grade top-k via cross-encoder score threshold; below threshold → rewrite-and-retry or sub-query, capped iterations (default 2). |
| Confidence estimation | `core/confidence.py` (new) | Combine dense score, cross-encoder score, groundedness (when available) into one 0-1 confidence; threshold → refusal. |
| Citation verification | `core/citation_check.py` (new) | Sentence-split answer, map each `[Source: ...]`-cited sentence to its chunk, cross-encoder-based entailment score, flag unsupported sentences. |
| Metadata filtering | `core/ingestion.py`, `core/chunking.py`, `core/vectorstore.py` (extend) | Derive `jurisdiction`/`doc_type` per source file (config-driven mapping, not inferred), store as chunk metadata, pass through to `search(..., where=...)`. |
| RAPTOR | `core/raptor.py` (new), `scripts/build_raptor_tree.py` (new) | Recursive clustering + LLM summarization at index time; tree persisted as JSON/pickle next to `vectorstore_data/`; retrieval-time tree search added as a retriever option. |
| Retrieval cache | `core/cache.py` (new) | In-memory LRU (`functools.lru_cache` or small custom dict+OrderedDict), keyed on `(normalized_query, filters, flags)`. |
| Metrics/observability | `core/metrics.py` (new), `evaluation/metrics.py` (extend) | Per-query stage timings, token counts, cost estimate, similarity distribution; new eval aggregate fields: recall@k, nDCG, latency, throughput, cost/query, context precision/recall, faithfulness, answer relevance, citation accuracy, hallucination rate. |
| FastAPI service | `api/main.py`, `api/schemas.py` (new) | Async wrapper around an async `QueryPipeline`; `/query` (SSE streaming) and `/query/sync` endpoints; concurrent retrieval via `asyncio.gather`. |
| Streamlit client | `ui/app.py` (rewrite) | Becomes an HTTP/SSE client of the FastAPI service instead of importing `QueryPipeline` directly; renders the new metrics panel. |

## Data flow additions

`SearchResult` gains no new required fields (parent text and metadata ride
in existing `section`/new optional fields); `RAGResponse` gains optional
`confidence: float`, `citations_verified: list[bool] | None`, and
`metrics: QueryMetrics | None` fields — all additive, existing consumers
(`evaluation/metrics.py`, current UI) keep working unchanged where they
don't opt into the new fields.

## Error handling

- Every new LLM-call feature (HyDE, query expansion, CRAG rewrite,
  confidence's groundedness leg, citation verification) catches
  `genai_errors.APIError`/timeout the same way `generation.py` already does
  and **degrades to skipping that stage** rather than failing the query —
  e.g. if query expansion's LLM call fails, fall back to single-query
  retrieval; if citation verification fails, return the answer unverified
  rather than dropping it.
- CRAG/agentic loop has a hard iteration cap (default 2) to prevent runaway
  re-retrieval on persistently low-confidence queries.
- FAISS/Qdrant backends raise the same way Chroma does on missing
  persist dir — no new empty-store special-casing.

## Testing

- Unit tests per new pure-function module (`fusion.py`, `compression.py`,
  `confidence.py`, `citation_check.py`) — deterministic inputs/outputs, no
  network calls.
- `BM25Retriever`, `FaissStore`, `QdrantStore` get the same smoke-test
  pattern already used in `core/retrieval.py`/`core/vectorstore.py`
  (`__main__` block indexing a small sample + assertions), plus pytest
  wrappers.
- Integration test: one end-to-end `QueryPipeline.ask()` call per major flag
  combination (hybrid on/off, expansion on/off, CRAG on/off) against the
  existing small eval sample, asserting it returns a well-formed
  `RAGResponse` without raising.
- `evaluation/metrics.py` extensions get unit tests for the new metric
  calculations (recall@k, nDCG) against hand-computed fixtures.
- FastAPI endpoints get `httpx.AsyncClient`-based tests (sync `/query/sync`
  path primarily, to avoid flaky SSE-stream assertions).

## Phased build order

1. Retrieval quality: BM25 + RRF, query expansion, metadata filtering.
2. Context shaping: parent-document retrieval, extractive compression.
3. Reliability: HyDE, merged CRAG/agentic loop, confidence estimation,
   citation verification.
4. Evaluation: new metrics in `evaluation/metrics.py`.
5. Infra: `VectorStore` Protocol + FAISS + Qdrant(local), retrieval cache,
   metrics/observability + Streamlit panel.
6. Production polish: FastAPI async service + SSE streaming, Streamlit
   rewritten as a thin client.
7. RAPTOR: offline tree-building script + tree-aware retrieval option.

Each phase should land as its own set of commits and pass its own tests
before the next phase starts, since later phases (FastAPI service, RAPTOR)
build on interfaces established earlier (`VectorStore` Protocol, metrics
dataclasses).
