# RAG Technical Doc Assistant

[![CI](https://github.com/Parakh20/rag-technical-doc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Parakh20/rag-technical-doc-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-style Retrieval-Augmented Generation system for question answering
over technical documents. It covers PDF ingestion, three chunking strategies,
dense or hybrid (dense + BM25) retrieval with cross-encoder reranking,
optional query expansion / HyDE / corrective retrieval / RAPTOR tree retrieval,
grounded Gemini generation with inline citations, citation verification and
confidence-based refusal. Everything is served through a FastAPI service with SSE
streaming and a Streamlit client, with an LLM-as-judge evaluation suite on top.
The demo corpus is real drone/UAS regulation (India's DGCA, the US FAA) plus
supporting arXiv research papers.

## Architecture

```
Query
  │
  ▼
(optional) query expansion / HyDE           core/query_transform.py
  │
  ▼
Dense retrieval (BAAI/bge-small-en-v1.5)    core/vectorstore*.py  (Chroma default; FAISS, Qdrant)
  + optional BM25, fused with RRF           core/bm25.py, core/fusion.py
  │
  ▼
Rerank (cross-encoder/ms-marco-MiniLM-L-6-v2) or MMR       core/retrieval.py
  (optional corrective re-retrieval, RAPTOR tree, parent-page expansion)
  │
  ▼
(optional) context compression              core/compression.py
  │
  ▼
Generate (gemini-3.5-flash, falls back to gemini-3.1-flash-lite on API errors;
system prompt enforces context-only answers)              core/generation.py
  │
  ▼
Cited answer ([Source: filename, Page X]) + confidence/refusal, citation checks,
per-query metrics, LRU response cache                     pipeline/query.py
```

Indexing side (`scripts/build_index.py`):

```
PDF corpus → DocumentLoader (PyMuPDF, header/footer strip, heading detect)
           → chunk_pages (recursive split, 400-600 tokens, default strategy)
           → EmbeddingModel (BGE-small, batched)
           → ChromaStore (persistent, upsert by deterministic chunk_id)
```

Serving: `api/main.py` exposes `GET /health`, `POST /query/sync` and an
SSE-streaming `POST /query`. `ui/app.py` is a thin Streamlit client of that API.

## Corpus

| Document | Why |
|---|---|
| DGCA CAR Section 3, Series X, Part I | India's official RPAS operations requirements (UIN, UAOP, NPNT) |
| FAA AC 107-2A | US sUAS operating rules (VLOS, altitude, certification) |
| 3 arXiv papers on RPAS/UAV airspace integration & reliability | Technical depth, and a stand-in for ICAO Doc 10019, which is paywalled with no legitimate free mirror |

This mix makes cross-document retrieval genuinely hard: India and US rules use
different terms for overlapping concepts (UIN/UAOP vs.
registration/remote-pilot-certificate), and dense, ambiguous regulatory language
stresses chunking and retrieval quality more than a FAQ-style corpus would.

`corpus/download.py` fetches the documents automatically and skips files
already on disk. The PDFs are not committed.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Parakh20/rag-technical-doc-assistant.git
cd rag-technical-doc-assistant
python3 -m venv venv
source venv/bin/activate
# CPU-only torch keeps the install small; drop the extra index for a CUDA build.
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cp .env.example .env    # then set GOOGLE_API_KEY
```

Build the index. This downloads the corpus and, on first run, the embedding and
reranker models from Hugging Face:

```bash
python scripts/build_index.py                   # default: recursive chunking
python scripts/build_index.py --strategy fixed  # or: semantic
```

Run the API, then the UI in a second terminal:

```bash
uvicorn api.main:app --port 8000
streamlit run ui/app.py
```

Or query from Python directly:

```python
from pipeline.query import QueryPipeline
response = QueryPipeline().ask("What is the maximum altitude for small UAS operations?")
print(response.answer)
```

### Configuration

| Variable | Used by | Notes |
|---|---|---|
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | generation, query transforms, RAPTOR, eval judge | required for anything that calls Gemini; see [`.env.example`](.env.example) |
| `RAG_API_BASE_URL` | `ui/app.py` | defaults to `http://127.0.0.1:8000` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `core/embeddings.py` | set to `1` automatically once the default models are cached; set `0` to force online |

Retrieval alone (building the index, `RetrieverWithReranker.retrieve`) needs no
API key.

## Evaluation

```bash
python scripts/run_eval.py                       # retrieval + generation + chunking comparison
python scripts/run_eval.py --no-groundedness     # skip LLM-judge calls (cheaper)
python scripts/run_eval.py --skip-chunking-compare
python scripts/benchmark_retrieval_configs.py    # retrieval-only comparison of configs
python scripts/build_raptor_tree.py              # one-off: build the RAPTOR tree (Gemini calls)
python scripts/benchmark_retrieval_configs.py --raptor-tree vectorstore_data/raptor_tree.json
```

`run_eval.py` writes `results/eval_results.csv` (per question) and
`results/eval_summary.txt` locally. Those files are gitignored.

The eval set (`evaluation/eval_set.py`, mirrored in `eval_set.json`) has 50
questions: 25 factual, 10 procedural, 5 cross-document, and 10 deliberately
unanswerable (out-of-corpus).

### Results from the author's runs

**Retrieval (all 40 answerable questions, no API cost):**

| Metric | Score |
|---|---|
| Hit@1 | 87.5% |
| Hit@3 | 90.0% |
| Hit@5 | 95.0% |
| MRR | 0.90 |
| Precision@5 | 86.0% |

**Chunking strategy comparison (Hit@5):** fixed-size 95.0%, recursive (default)
95.0%, semantic 95.0%. All three tie on this corpus, so there's no quality
reason to pay semantic chunking's extra embedding cost on this kind of
regulatory text.

**Retrieval configurations** (`results/retrieval_config_comparison.json`):

| Config | Hit@1 | Hit@5 | MRR | Avg s/query |
|---|---|---|---|---|
| baseline (dense + rerank) | 0.875 | 0.950 | 0.899 | 1.2 |
| hybrid (dense + BM25) | 0.875 | 0.925 | 0.890 | 1.8 |
| query expansion | 0.875 | 0.950 | 0.895 | 38.0 |
| HyDE | 0.875 | 0.950 | 0.901 | 40.2 |
| CRAG correction | 0.875 | 0.950 | 0.899 | 2.0 |
| RAPTOR | 0.350 | 0.925 | 0.621 | 2.1 |

The LLM-assisted configs mostly add latency without moving retrieval quality on
this corpus. The per-query latency of query expansion and HyDE is dominated by
the free-tier rate limiter in `core/generation.py`.

**Generation quality (`gemini-3.5-flash` → `gemini-3.1-flash-lite` fallback):**

| Metric | Score |
|---|---|
| Groundedness score | 0.96 / 1.0 |
| Answer rate | 77.5% (of 40 answerable questions) |
| Refusal rate | 50.0% (of 10 unanswerable questions; see note) |

> **Note on completion:** 42 of 50 questions got a real model response; 8
> failed on transient network errors during that run (not model-side failures).
> Of the 10 unanswerable questions, 8 got a real response and 3 of those 8 were
> genuinely refused (37.5% true refusal accuracy on real responses). The
> reported 50% refusal rate also counts the 2 failed-to-respond unanswerable
> questions as "refused", which inflates it relative to the true 37.5%.

## Running the tests

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
python -m pytest -q
```

The suite runs offline with no API keys and no model downloads. A deterministic
fake embedder and cross-encoder (`tests/conftest.py`) stand in for the
sentence-transformers models, and the Gemini client is mocked. The tests cover
ingestion, chunking, the Chroma/FAISS/Qdrant stores, BM25, fusion, reranking,
MMR, compression, citation verification, confidence, caching, `QueryPipeline`
and the FastAPI endpoints, including SSE streaming. CI runs them on every push
and pull request.

## Project structure

```
api/          FastAPI service (sync + SSE streaming endpoints) and request/response schemas
core/         Ingestion, chunking, embeddings, vector stores, BM25, fusion, retrieval,
              query transforms, RAPTOR, compression, generation, citation checks,
              confidence, cache, metrics
pipeline/     Indexer and end-to-end QueryPipeline
corpus/       Corpus downloader (PDFs are fetched, not committed)
evaluation/   Eval set and retrieval/generation/groundedness metrics
scripts/      CLI entry points: build_index, run_eval, benchmark_retrieval_configs, build_raptor_tree
ui/           Streamlit client of the API
tests/        pytest suite (offline, fakes for models and the LLM)
docs/design/  Design notes for the advanced-RAG upgrade
results/      Committed retrieval-config comparison; local eval outputs are gitignored
```

## Operational notes

- `core/embeddings.py` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before
  importing sentence-transformers once the default models are cached. Without
  this, huggingface_hub makes a "check for newer files" network call even for a
  cached model, and on an unreliable network that call can hang for minutes.
  On a fresh machine it stays online so the first run can download the weights.
- `evaluation/metrics.py` wraps each generation/judge call in a hard
  daemon-thread timeout (`GENERATION_TIMEOUT_SECONDS`), so the eval loop keeps
  moving even when an API client's own timeout isn't honored against a
  half-dead socket.
- `core/generation.py` paces every Gemini call (`MIN_SECONDS_BETWEEN_CALLS`) to
  stay under the free tier's 15 requests/minute, retries 429s using the API's
  suggested delay, and sets an explicit request timeout so an overloaded model
  raises a retriable error instead of hanging.

## Key technical decisions

- **Two-stage retrieval (dense + cross-encoder reranking).** Dense search is
  fast but approximate. A cross-encoder scores each (query, chunk) pair
  directly and is far more precise, but too slow to run over the whole corpus,
  so it only reranks the dense stage's top 20 candidates.
- **Hybrid retrieval with RRF.** BM25 catches exact-match signals that
  embeddings can blur (regulation IDs, abbreviations like UIN/VLOS, numbers
  like 400 feet). Reciprocal rank fusion merges it with dense results without
  needing comparable score scales.
- **MMR as an alternative to reranking**, for when context diversity matters
  more than pure relevance (`lambda_mult`, default 0.7).
- **Swappable vector stores.** Chroma, FAISS and Qdrant all implement the
  `VectorStore` protocol in `core/vectorstore.py`, so retrieval code doesn't
  depend on a specific backend.
- **Groundedness via LLM-as-judge**, plus sentence-level citation verification
  with the cross-encoder, rather than relying only on refusal-phrase matching.
- **Three chunking strategies, compared empirically** rather than assumed.
- **Deterministic chunk IDs** (`uuid5` over source + strategy + index), so
  re-running the indexer is an idempotent upsert instead of piling up
  duplicates.

## License

[MIT](LICENSE)
