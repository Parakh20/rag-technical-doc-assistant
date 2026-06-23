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

**Generation quality (partial - see note):**

| Metric | Score |
|---|---|
| Groundedness score | 0.97 / 1.0 (mean over 20 successfully-judged answers) |
| Answer rate | 47.5% (raw aggregate, partial run) |
| Refusal rate | 100% (artifact, not a real measurement - see note) |

> **Note on completion:** the evaluator account's credit balance was
> exhausted partway through the 50-call generation run. 21 of 40
> answerable questions got a real generation (19 of 21 grounded, mean
> groundedness 0.97 on those that succeeded); the remaining answerable
> questions and **all 10 unanswerable questions** failed with "credit
> balance too low" before getting a model response. The refusal_rate
> figure above is therefore not a real measurement of refusal behavior -
> it reflects zero successful unanswerable-question generations, not
> correct refusals. Re-run `python scripts/run_eval.py` with a
> sufficient balance for a complete, trustworthy generation-quality
> number. Full per-question detail (including which rows are real vs.
> `<GENERATION FAILED: ...>`) is in `results/eval_results.csv`; the raw
> annotated summary is in `results/eval_summary.txt`.

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
  daemon-thread timeout (`GENERATION_TIMEOUT_SECONDS`). The Anthropic
  client's own `timeout=` config isn't reliably honored against a half-dead
  socket in every environment; the daemon-thread wrapper guarantees the
  eval loop keeps moving regardless of what the underlying connection is
  doing, at the cost of occasionally abandoning an orphaned thread.

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
