"""Run the full evaluation suite: retrieval metrics, generation quality
(groundedness via LLM-as-judge), and chunking strategy comparison.

Writes results/eval_results.csv and results/eval_summary.txt.

Usage: python scripts/run_eval.py [--no-groundedness] [--skip-chunking-compare]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    RESULTS_DIR,
    aggregate_metrics,
    compare_chunking_strategies,
    run_evaluation,
)


def write_csv(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].to_dict().keys()) if results else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def write_summary(agg: dict, chunking_comparison: dict | None, path: Path) -> None:
    lines = [
        "RETRIEVAL PERFORMANCE",
        f"  Hit@1:  {agg['hit_at_1']*100:.1f}%",
        f"  Hit@3:  {agg['hit_at_3']*100:.1f}%",
        f"  Hit@5:  {agg['hit_at_5']*100:.1f}%",
        f"  MRR:    {agg['mrr']:.2f}",
        f"  Precision@5: {agg['precision_at_5']*100:.1f}%",
        f"  Recall@5:    {agg['recall_at_5']*100:.1f}%",
        f"  nDCG@5:      {agg['ndcg_at_5']:.3f}",
        "",
        "GENERATION QUALITY",
        f"  Groundedness score:  {agg['groundedness_score']:.2f} / 1.0",
        f"  Answer rate:         {agg['answer_rate']*100:.1f}% (of {agg['num_answerable']} answerable questions)",
        f"  Refusal rate:        {agg['refusal_rate']*100:.1f}% (of {agg['num_unanswerable']} unanswerable questions)",
        "",
        "RAGAS-STYLE / OBSERVABILITY",
        f"  Faithfulness:        {agg['faithfulness']:.2f}",
        f"  Answer relevance:    {agg['answer_relevance']:.2f}",
        f"  Context precision:   {agg['context_precision']*100:.1f}%",
        f"  Context recall:      {agg['context_recall']*100:.1f}%",
        f"  Hallucination rate:  {agg['hallucination_rate']:.2f}",
        f"  Avg latency/query:   {agg['avg_latency_seconds']:.2f}s",
        f"  Throughput:          {agg['throughput_qps']:.3f} q/s",
        f"  Avg cost/query:      ${agg['avg_cost_usd']:.5f} (rough estimate, not billed truth)",
    ]
    if chunking_comparison:
        lines += [
            "",
            "CHUNKING STRATEGY COMPARISON (Hit@5)",
            f"  Fixed-size:          {chunking_comparison.get('fixed', 0)*100:.1f}%",
            f"  Recursive split:     {chunking_comparison.get('recursive', 0)*100:.1f}%",
            f"  Semantic:            {chunking_comparison.get('semantic', 0)*100:.1f}%",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-groundedness", action="store_true",
                         help="Skip LLM-as-judge groundedness scoring (saves API calls)")
    parser.add_argument("--skip-chunking-compare", action="store_true",
                         help="Skip the fixed/recursive/semantic chunking comparison")
    parser.add_argument("--hybrid", action="store_true",
                         help="Enable hybrid (BM25+dense, RRF) retrieval for this run")
    parser.add_argument("--output-prefix", default="eval",
                         help="Output file prefix under results/ (default: eval -> eval_results.csv/eval_summary.txt)")
    args = parser.parse_args()

    retrieval_kwargs = {"use_hybrid": True} if args.hybrid else None
    print(f"=== Running evaluation suite (50 questions{', hybrid retrieval' if args.hybrid else ''}) ===")
    results = run_evaluation(
        score_groundedness_enabled=not args.no_groundedness, retrieval_kwargs=retrieval_kwargs
    )
    write_csv(results, RESULTS_DIR / f"{args.output_prefix}_results.csv")

    agg = aggregate_metrics(results)

    chunking_comparison = None
    if not args.skip_chunking_compare:
        print("\n=== Comparing chunking strategies (retrieval-only, Hit@5) ===")
        chunking_comparison = compare_chunking_strategies()

    write_summary(agg, chunking_comparison, RESULTS_DIR / f"{args.output_prefix}_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
