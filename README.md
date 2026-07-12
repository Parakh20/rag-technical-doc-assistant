# RAG Technical Doc Assistant

A production-style Retrieval-Augmented Generation pipeline for technical
document Q&A: PDF ingestion, three chunking strategies, dense + cross-encoder
reranked retrieval, Gemini-powered grounded generation with citations, and an
LLM-as-judge evaluation suite. Demoed against real drone/UAS regulatory
documents (DGCA, FAA) plus supporting research papers.

## Architecture

```
Query
  │
  ▼
Embed (BAAI/bge-small-en-v1.5)
  │
  ▼
Dense Retrieval (Chroma, top-20)
  │
  ▼
Rerank (cross-encoder/ms-marco-MiniLM-L-6-v2, top-5)   [or MMR for diversity]
  │
  ▼
Generate (gemini-3.5-flash, falls back to gemini-3.1-flash-lite on 429/503;
system prompt enforces context-only answers)
  │
  ▼
Cited Answer  ([Source: filename, Page X] inline)
```

Indexing side (`scripts/build_index.py`):
```
PDF corpus → DocumentLoader (PyMuPDF, header/footer strip, heading detect)
           → chunk_pages (recursive split, 400-600 tokens, default strategy)
           → EmbeddingModel (BGE-small, batched)
           → ChromaStore (persistent, upsert by deterministic chunk_id)
```

## Corpus

| Document | Why |
|---|---|
| DGCA CAR Section 3, Series X, Part I | India's official RPAS operations requirements (UIN, UAOP, NPNT) |
| FAA AC 107-2A | US sUAS operating rules (VLOS, altitude, certification) |
| 3 arXiv papers on RPAS/UAV airspace integration & reliability | Technical depth + a substitute for ICAO Doc 10019, which is paywalled on the ICAO Store with no legitimate free mirror (the official icao.int link 403s) |

This combination gives genuine cross-document retrieval challenges (India vs.
US rules use different terminology for overlapping concepts - UIN/UAOP vs.
registration/remote-pilot-certificate) and dense, ambiguous regulatory
language that stresses chunking and retrieval quality more than a typical
FAQ-style corpus would.

`corpus/download.py` fetches these automatically (idempotent - re-running
skips files already on disk).

## Evaluation Results

50-question eval set: 25 factual, 10 procedural, 5 cross-document, 10
deliberately unanswerable (out-of-corpus) questions.

**Retrieval (full run, all 40 answerable questions, no API cost):**

| Metric | Score |
|---|---|
| Hit@1 | 87.5% |
| Hit@3 | 90.0% |
| Hit@5 | 95.0% |
| MRR | 0.90 |
| Precision@5 | 86.0% |

**Chunking strategy comparison (Hit@5, full run, no API cost):**

| Strategy | Hit@5 |
|---|---|
| Fixed-size | 95.0% |
| Recursive split (default) | 95.0% |
| Semantic | 95.0% |

All three tie on this corpus - the cheap default (recursive splitting)
performs as well as the more expensive semantic strategy here, so there's
no quality reason to pay the extra embedding cost for semantic chunking on
this kind of regulatory/technical text.

**Generation quality (Gemini, `gemini-3.5-flash` -> `gemini-3.1-flash-lite` fallback):**

| Metric | Score |
|---|---|
| Groundedness score | 0.96 / 1.0 |
| Answer rate | 77.5% (of 40 answerable questions) |
| Refusal rate | 50.0% (of 10 unanswerable questions; see note) |

> **Note on completion:** 42 of 50 questions got a real model response;
> 8 failed on transient network errors in this sandbox (DNS blips, one
> network-unreachable, one read timeout - not model-side failures). Of
> the 10 unanswerable questions, 8 got a real response and 3 of those 8
> were genuinely refused (37.5% true refusal accuracy on real
> responses). The reported 50% refusal_rate also counts the 2
> failed-to-respond unanswerable questions as "refused" by default,
> which inflates it slightly versus the true 37.5% figure. Full
> per-question detail is in `results/eval_results.csv`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):
```
GOOGLE_API_KEY=...
```

## Usage

Build the index (downloads the corpus on first run):
```bash
python scripts/build_index.py                  # default: recursive chunking
python scripts/build_index.py --strategy fixed  # or: semantic
```

Run the chat UI:
```bash
streamlit run ui/app.py
```

Run the evaluation suite:
```bash
python scripts/run_eval.py                      # full run: retrieval + generation + chunking comparison
python scripts/run_eval.py --no-groundedness     # skip LLM-judge calls (cheaper)
python scripts/run_eval.py --skip-chunking-compare
```

Query from Python directly:
```python
from pipeline.query import QueryPipeline
response = QueryPipeline().ask("What is the maximum altitude for small UAS operations?")
print(response.answer)
```

## Operational notes

- `core/embeddings.py` and `core/retrieval.py` force `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` before importing sentence-transformers. Without
  this, huggingface_hub makes a "check for newer files" network call even
  when the model is already cached, and on an unreliable network that call
  can hang for many minutes instead of failing fast.
- `evaluation/metrics.py` wraps each generation/judge call in a hard
  daemon-thread timeout (`GENERATION_TIMEOUT_SECONDS`). API client
  `timeout=` config isn't always reliably honored against a half-dead
  socket in every environment; the daemon-thread wrapper guarantees the
  eval loop keeps moving regardless of what the underlying connection is
  doing, at the cost of occasionally abandoning an orphaned thread.
- `core/generation.py` paces every Gemini API call (`MIN_SECONDS_BETWEEN_CALLS`)
  and retries 429s using the API's own suggested delay. The free tier caps
  `gemini-3.1-flash-lite` at 15 requests/minute; without pacing, a
  sustained eval run bursts past that immediately. The `genai.Client`
  also gets an explicit `http_options.timeout` - without it, a call to an
  overloaded model can hang indefinitely instead of raising a fast,
  retriable error.

## Key technical decisions

- **Two-stage retrieval (dense + cross-encoder reranking).** Dense
  embedding search is fast but approximate; a cross-encoder scores the
  (query, chunk) pair directly and is far more precise, but too slow to run
  over the whole corpus. Running it only over the dense stage's top-20
  candidates gets near cross-encoder quality at a fraction of the cost.
- **MMR as an alternative to reranking**, for cases where you want
  context diversity instead of pure relevance ranking (`lambda_mult`
  controls the relevance/diversity tradeoff, default 0.7).
- **Groundedness via LLM-as-judge.** Rather than relying solely on
  string-matching refusal phrases, a second Gemini call scores whether the
  generated answer's claims are actually supported by the retrieved context.
- **Three chunking strategies, compared empirically** rather than assumed:
  fixed-size token windows, recursive paragraph/sentence/word splitting
  (the default), and embedding-similarity-based semantic chunking. See the
  comparison table above for which wins on this corpus.
- **Deterministic chunk IDs** (`uuid5` over source+strategy+index) so that
  re-running the indexer is a true idempotent upsert instead of accumulating
  duplicate chunks on every rebuild.
