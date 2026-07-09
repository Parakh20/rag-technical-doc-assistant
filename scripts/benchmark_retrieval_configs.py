"""Retrieval-only benchmark across retrieval configurations (baseline,
hybrid, query expansion, HyDE, CRAG correction, RAPTOR) - no generation or
groundedness judge calls, so this is cheap even for configs that add an
LLM call during retrieval itself (query expansion, HyDE). Mirrors the
existing chunking-strategy comparison (evaluation/metrics.py
compare_chunking_strategies) in spirit: measure retrieval quality directly
rather than paying for full generation to get a retrieval-quality signal.

Usage: python scripts/benchmark_retrieval_configs.py [--raptor-tree PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import RetrieverWithReranker
from core.vectorstore import ChromaStore
from evaluation.eval_set import EVAL_QUESTIONS
from evaluation.metrics import RESULTS_DIR, retrieval_metrics_for_question


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(per_question_metrics: list[dict]) -> dict:
    return {
        "hit_at_1": _safe_mean([1.0 if m["hit_at_1"] else 0.0 for m in per_question_metrics]),
        "hit_at_3": _safe_mean([1.0 if m["hit_at_3"] else 0.0 for m in per_question_metrics]),
        "hit_at_5": _safe_mean([1.0 if m["hit_at_5"] else 0.0 for m in per_question_metrics]),
        "mrr": _safe_mean([m["reciprocal_rank"] for m in per_question_metrics]),
        "precision_at_5": _safe_mean([m["precision_at_5"] for m in per_question_metrics]),
        "recall_at_5": _safe_mean([1.0 if m["recall_at_5"] else 0.0 for m in per_question_metrics]),
        "ndcg_at_5": _safe_mean([m["ndcg_at_5"] for m in per_question_metrics]),
    }


def benchmark_config(
    retriever: RetrieverWithReranker, questions: list[dict], retrieval_kwargs: dict
) -> dict:
    per_question_metrics = []
    start = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {q['id']}...", flush=True)
        metrics, _chunks = retrieval_metrics_for_question(retriever, q, retrieval_kwargs=retrieval_kwargs)
        per_question_metrics.append(metrics)
    elapsed = time.monotonic() - start
    agg = _aggregate(per_question_metrics)
    agg["avg_seconds_per_query"] = elapsed / len(questions) if questions else 0.0
    return agg


def benchmark_correction(retriever: RetrieverWithReranker, questions: list[dict]) -> dict:
    """retrieve_with_correction has a different return shape (results, iterations),
    so it can't go through retrieval_metrics_for_question's retrieval_kwargs path."""
    from evaluation.metrics import math

    per_question_metrics = []
    total_iterations = 0
    start = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {q['id']}...", flush=True)
        results, iterations = retriever.retrieve_with_correction(q["question"], dense_k=20, final_k=5)
        total_iterations += iterations
        expected = q["expected_source"]
        hit_ranks = [j + 1 for j, r in enumerate(results) if r.source == expected]
        precision_at_5 = sum(1 for r in results if r.source == expected) / len(results) if results else 0.0
        hit_at_5 = any(r <= 5 for r in hit_ranks)
        per_question_metrics.append({
            "hit_at_1": 1 in hit_ranks, "hit_at_3": any(r <= 3 for r in hit_ranks),
            "hit_at_5": hit_at_5, "reciprocal_rank": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
            "precision_at_5": precision_at_5, "recall_at_5": hit_at_5,
            "ndcg_at_5": 1.0 / math.log2(hit_ranks[0] + 1) if hit_ranks else 0.0,
        })
    elapsed = time.monotonic() - start
    agg = _aggregate(per_question_metrics)
    agg["avg_seconds_per_query"] = elapsed / len(questions) if questions else 0.0
    agg["avg_iterations"] = total_iterations / len(questions) if questions else 0.0
    return agg


def benchmark_raptor(retriever: RetrieverWithReranker, questions: list[dict], tree_path: str) -> dict:
    per_question_metrics = []
    start = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {q['id']}...", flush=True)
        results = retriever.retrieve_raptor(q["question"], tree_path, dense_k=20, final_k=5)
        expected = q["expected_source"]
        hit_ranks = [j + 1 for j, r in enumerate(results) if r.source == expected]
        precision_at_5 = sum(1 for r in results if r.source == expected) / len(results) if results else 0.0
        hit_at_5 = any(r <= 5 for r in hit_ranks)
        per_question_metrics.append({
            "hit_at_1": 1 in hit_ranks, "hit_at_3": any(r <= 3 for r in hit_ranks),
            "hit_at_5": hit_at_5, "reciprocal_rank": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
            "precision_at_5": precision_at_5, "recall_at_5": hit_at_5,
            "ndcg_at_5": __import__("math").log2(2) / __import__("math").log2(hit_ranks[0] + 1) if hit_ranks else 0.0,
        })
    elapsed = time.monotonic() - start
    agg = _aggregate(per_question_metrics)
    agg["avg_seconds_per_query"] = elapsed / len(questions) if questions else 0.0
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raptor-tree", type=Path, default=None,
                         help="Path to a pre-built RAPTOR tree JSON (skip RAPTOR config if omitted)")
    parser.add_argument("--configs", nargs="+",
                         default=["baseline", "hybrid", "query_expansion", "hyde", "crag"],
                         help="Which configs to run (raptor added automatically if --raptor-tree given)")
    args = parser.parse_args()

    store = ChromaStore()
    retriever = RetrieverWithReranker(store)
    answerable = [q for q in EVAL_QUESTIONS if q["expected_source"] is not None]
    print(f"Benchmarking retrieval configs over {len(answerable)} answerable questions "
          f"(full 50-question eval set's answerable subset)")

    results: dict[str, dict] = {}
    config_kwargs = {
        "baseline": {},
        "hybrid": {"use_hybrid": True},
        "query_expansion": {"use_query_expansion": True},
        "hyde": {"use_hyde": True},
    }

    for name in args.configs:
        if name == "crag":
            print(f"\n=== {name} ===")
            results[name] = benchmark_correction(retriever, answerable)
        elif name in config_kwargs:
            print(f"\n=== {name} ===")
            results[name] = benchmark_config(retriever, answerable, config_kwargs[name])
        else:
            print(f"[skip] unknown config {name!r}")

    if args.raptor_tree:
        print("\n=== raptor ===")
        results["raptor"] = benchmark_raptor(retriever, answerable, str(args.raptor_tree))

    output_path = RESULTS_DIR / "retrieval_config_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        # Merge rather than clobber - a re-run with --configs raptor (say) shouldn't
        # erase results from an earlier run that covered other configs.
        existing = json.loads(output_path.read_text())
        existing.update(results)
        results = existing
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
