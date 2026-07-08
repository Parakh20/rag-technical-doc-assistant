"""Confidence estimation: combine retrieval signals into one 0-1 score used
to decide whether the pipeline should answer or explicitly refuse.

Cross-encoder scores are raw ms-marco logits (unbounded, roughly [-11, 11]
on this corpus), not probabilities - a sigmoid squashes them into [0, 1]
before averaging with the (already [0, 1]) dense cosine similarity and the
(already [0, 1]) LLM-judge groundedness score.
"""

from __future__ import annotations

import math

from core.vectorstore import SearchResult

DEFAULT_REFUSAL_THRESHOLD = 0.35


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_confidence(
    top_chunk: SearchResult | None, groundedness_score: float | None = None
) -> float:
    """0-1 confidence for a query's retrieval (+ optionally generation)."""
    if top_chunk is None:
        return 0.0
    cross_encoder_prob = _sigmoid(top_chunk.score)
    components = [top_chunk.dense_score, cross_encoder_prob]
    if groundedness_score is not None:
        components.append(groundedness_score)
    return sum(components) / len(components)


def should_refuse(confidence: float, threshold: float = DEFAULT_REFUSAL_THRESHOLD) -> bool:
    return confidence < threshold


if __name__ == "__main__":
    strong = SearchResult(
        text="x", source="s", page=1, section="", chunk_id="1",
        score=5.0, dense_score=0.85,
    )
    weak = SearchResult(
        text="x", source="s", page=1, section="", chunk_id="2",
        score=-8.0, dense_score=0.3,
    )
    print("strong confidence:", compute_confidence(strong), "refuse:", should_refuse(compute_confidence(strong)))
    print("weak confidence:", compute_confidence(weak), "refuse:", should_refuse(compute_confidence(weak)))
    print("no chunk confidence:", compute_confidence(None), "refuse:", should_refuse(compute_confidence(None)))
    assert not should_refuse(compute_confidence(strong))
    assert should_refuse(compute_confidence(weak))
    print("OK")
