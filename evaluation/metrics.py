"""Retrieval + generation evaluation metrics, and chunking-strategy comparison."""

from __future__ import annotations

import queue
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

GENERATION_TIMEOUT_SECONDS = 150  # generous margin over the 429 retry-with-backoff budget


def _call_with_hard_timeout(fn, args, timeout: float):
    """Run fn(*args) on a daemon thread and wait up to `timeout` seconds.

    A plain function call has no reliable way to be interrupted once a
    blocking network read is stuck (httpx's own timeout isn't always
    honored against a half-dead socket in this environment). Running it on
    a daemon thread means: if it times out, we return control to the
    caller immediately and the orphaned thread is abandoned without
    blocking process exit (daemon threads are not joined at interpreter
    shutdown, unlike ThreadPoolExecutor workers).
    """
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _target():
        try:
            result_queue.put(("ok", fn(*args)))
        except Exception as exc:  # noqa: BLE001 - forward to caller's thread
            result_queue.put(("error", exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        status, value = result_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(
            f"call exceeded {timeout}s (network stall - moving on, "
            "orphaned thread abandoned in background)"
        ) from None
    if status == "error":
        raise value
    return value

from core.chunking import chunk_pages
from core.generation import RAGGenerator
from core.ingestion import DocumentLoader
from core.retrieval import RetrieverWithReranker
from core.vectorstore import ChromaStore, SearchResult
from evaluation.eval_set import EVAL_QUESTIONS

RESULTS_DIR = Path(__file__).parent.parent / "results"
CORPUS_DIR = Path(__file__).parent.parent / "corpus"

GROUNDEDNESS_JUDGE_PROMPT = """You are evaluating whether an AI-generated answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Does this answer contain any claims NOT supported by the provided context? \
Respond with ONLY a single number between 0 and 1, where 1.0 means fully \
grounded (every claim is supported by the context) and 0.0 means entirely \
unsupported/fabricated. Output only the number, nothing else."""


@dataclass
class QuestionResult:
    id: str
    question: str
    question_type: str
    expected_source: str | None
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    reciprocal_rank: float | None
    precision_at_5: float | None
    answer: str
    grounded: bool
    groundedness_score: float | None
    chunks_used: int
    tokens_used: int

    def to_dict(self) -> dict:
        return asdict(self)


def retrieval_metrics_for_question(
    retriever: RetrieverWithReranker, question: dict, k_max: int = 5
) -> tuple[dict, list[SearchResult]]:
    """Hit@1/3/5, reciprocal rank, precision@5 for one question.
    Returns (metrics_dict, retrieved_chunks)."""
    results = retriever.retrieve(question["question"], dense_k=20, final_k=k_max)
    expected = question["expected_source"]

    if expected is None:
        return (
            {"hit_at_1": None, "hit_at_3": None, "hit_at_5": None,
             "reciprocal_rank": None, "precision_at_5": None},
            results,
        )

    hit_ranks = [i + 1 for i, r in enumerate(results) if r.source == expected]
    precision_at_5 = (
        sum(1 for r in results if r.source == expected) / len(results) if results else 0.0
    )
    metrics = {
        "hit_at_1": 1 in hit_ranks,
        "hit_at_3": any(r <= 3 for r in hit_ranks),
        "hit_at_5": any(r <= 5 for r in hit_ranks),
        "reciprocal_rank": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
        "precision_at_5": precision_at_5,
    }
    return metrics, results


def score_groundedness(
    generator: RAGGenerator, question: str, answer: str, chunks: list[SearchResult]
) -> float:
    """LLM-as-judge groundedness score in [0, 1]."""
    from google.genai import errors as genai_errors
    from google.genai import types

    from core.generation import _call_with_rate_limit_retry

    context = "\n\n".join(f"[CHUNK {i + 1}] {c.text}" for i, c in enumerate(chunks))
    prompt = GROUNDEDNESS_JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    config = types.GenerateContentConfig(max_output_tokens=10)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    try:
        response = _call_with_rate_limit_retry(
            generator.client.models.generate_content,
            model=generator.model, contents=contents, config=config,
        )
    except genai_errors.APIError:
        response = _call_with_rate_limit_retry(
            generator.client.models.generate_content,
            model=generator.fallback_model, contents=contents, config=config,
        )
    text = (response.text or "").strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return 0.5  # judge returned something unparseable; neutral fallback


def run_evaluation(
    store: ChromaStore | None = None,
    generator: RAGGenerator | None = None,
    score_groundedness_enabled: bool = True,
    questions: list[dict] | None = None,
) -> list[QuestionResult]:
    store = store or ChromaStore()
    retriever = RetrieverWithReranker(store)
    generator = generator or RAGGenerator()
    questions = questions if questions is not None else EVAL_QUESTIONS

    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}...", flush=True)
        retrieval_metrics, chunks = retrieval_metrics_for_question(retriever, q)

        try:
            response = _call_with_hard_timeout(
                generator.generate, (q["question"], chunks), GENERATION_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating
            print(f"  [error] {q['id']}: generation failed ({exc}); recording as failed question")
            results.append(
                QuestionResult(
                    id=q["id"], question=q["question"], question_type=q["question_type"],
                    expected_source=q["expected_source"],
                    hit_at_1=retrieval_metrics["hit_at_1"], hit_at_3=retrieval_metrics["hit_at_3"],
                    hit_at_5=retrieval_metrics["hit_at_5"], reciprocal_rank=retrieval_metrics["reciprocal_rank"],
                    precision_at_5=retrieval_metrics["precision_at_5"],
                    answer=f"<GENERATION FAILED: {exc}>", grounded=False, groundedness_score=None,
                    chunks_used=len(chunks), tokens_used=0,
                )
            )
            continue

        groundedness_score = None
        if score_groundedness_enabled and chunks:
            try:
                groundedness_score = _call_with_hard_timeout(
                    score_groundedness, (generator, q["question"], response.answer, chunks),
                    GENERATION_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001 - judge failure shouldn't drop the question
                print(f"  [error] {q['id']}: groundedness judge failed ({exc})")

        results.append(
            QuestionResult(
                id=q["id"],
                question=q["question"],
                question_type=q["question_type"],
                expected_source=q["expected_source"],
                hit_at_1=retrieval_metrics["hit_at_1"],
                hit_at_3=retrieval_metrics["hit_at_3"],
                hit_at_5=retrieval_metrics["hit_at_5"],
                reciprocal_rank=retrieval_metrics["reciprocal_rank"],
                precision_at_5=retrieval_metrics["precision_at_5"],
                answer=response.answer,
                grounded=response.grounded,
                groundedness_score=groundedness_score,
                chunks_used=response.chunks_used,
                tokens_used=response.tokens_used,
            )
        )
    return results


def aggregate_metrics(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.expected_source is not None]
    unanswerable = [r for r in results if r.expected_source is None]

    def safe_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    groundedness_scores = [r.groundedness_score for r in results if r.groundedness_score is not None]

    return {
        "hit_at_1": safe_mean([1.0 if r.hit_at_1 else 0.0 for r in answerable]),
        "hit_at_3": safe_mean([1.0 if r.hit_at_3 else 0.0 for r in answerable]),
        "hit_at_5": safe_mean([1.0 if r.hit_at_5 else 0.0 for r in answerable]),
        "mrr": safe_mean([r.reciprocal_rank for r in answerable]),
        "precision_at_5": safe_mean([r.precision_at_5 for r in answerable]),
        "groundedness_score": safe_mean(groundedness_scores),
        "answer_rate": safe_mean([1.0 if r.grounded else 0.0 for r in answerable]),
        "refusal_rate": safe_mean([1.0 if not r.grounded else 0.0 for r in unanswerable]),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
    }


def compare_chunking_strategies(
    strategies: tuple[str, ...] = ("fixed", "recursive", "semantic"),
    sample_size: int | None = None,
) -> dict[str, float]:
    """Build a temporary index per chunking strategy and report Hit@5
    over the answerable questions (retrieval-only, no generation cost)."""
    answerable = [q for q in EVAL_QUESTIONS if q["expected_source"] is not None]
    if sample_size:
        answerable = answerable[:sample_size]

    pages = DocumentLoader().load_directory(CORPUS_DIR)
    hit_at_5_by_strategy: dict[str, float] = {}

    for strategy in strategies:
        persist_dir = RESULTS_DIR / f"_chunking_compare_{strategy}"
        shutil.rmtree(persist_dir, ignore_errors=True)
        chunks = chunk_pages(pages, strategy=strategy)
        store = ChromaStore(persist_dir=persist_dir, collection_name=f"compare_{strategy}")
        store.add_documents(chunks)
        retriever = RetrieverWithReranker(store)

        hits = 0
        for q in answerable:
            results = retriever.retrieve(q["question"], dense_k=20, final_k=5, use_reranker=False)
            if any(r.source == q["expected_source"] for r in results):
                hits += 1
        hit_at_5_by_strategy[strategy] = hits / len(answerable) if answerable else 0.0
        shutil.rmtree(persist_dir, ignore_errors=True)

    return hit_at_5_by_strategy
