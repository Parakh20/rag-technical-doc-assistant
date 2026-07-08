"""Reciprocal Rank Fusion: merge multiple ranked result lists (e.g. BM25 +
dense, or dense results for several query reformulations) into one ranking,
without needing the lists' raw scores to be on comparable scales.

score(doc) = sum over lists containing doc of  1 / (k + rank_in_that_list)
"""

from __future__ import annotations

from core.vectorstore import SearchResult

DEFAULT_RRF_K = 60  # standard constant from the original RRF paper


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = DEFAULT_RRF_K
) -> list[SearchResult]:
    """Merge ranked lists by chunk_id, summing 1/(k+rank) across lists.
    First occurrence of a chunk_id (by fused score) wins for the returned
    SearchResult's text/metadata."""
    fused_scores: dict[str, float] = {}
    representative: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            representative.setdefault(result.chunk_id, result)

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        SearchResult(
            text=representative[cid].text,
            source=representative[cid].source,
            page=representative[cid].page,
            section=representative[cid].section,
            chunk_id=cid,
            score=fused_scores[cid],
            jurisdiction=representative[cid].jurisdiction,
            doc_type=representative[cid].doc_type,
            parent_id=representative[cid].parent_id,
            parent_text=representative[cid].parent_text,
            dense_score=representative[cid].dense_score,
        )
        for cid in ordered_ids
    ]


if __name__ == "__main__":
    a = [
        SearchResult(text="a", source="s", page=1, section="", chunk_id="1", score=0.9),
        SearchResult(text="b", source="s", page=1, section="", chunk_id="2", score=0.8),
    ]
    b = [
        SearchResult(text="b", source="s", page=1, section="", chunk_id="2", score=5.0),
        SearchResult(text="c", source="s", page=1, section="", chunk_id="3", score=4.0),
    ]
    fused = reciprocal_rank_fusion([a, b])
    for r in fused:
        print(f"  chunk_id={r.chunk_id} fused_score={r.score:.4f}")
    assert fused[0].chunk_id == "2", "doc ranked in both lists should come first"
    print("OK")
