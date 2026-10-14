import math

import pytest

from core.confidence import compute_confidence, should_refuse
from core.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from tests.conftest import make_result


def test_rrf_ranks_documents_present_in_both_lists_first():
    dense = [make_result("1"), make_result("2")]
    lexical = [make_result("2"), make_result("3")]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert [r.chunk_id for r in fused] == ["2", "1", "3"]
    assert fused[0].score == pytest.approx(1 / (DEFAULT_RRF_K + 2) + 1 / (DEFAULT_RRF_K + 1))


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_keeps_metadata_of_first_occurrence():
    first = make_result("x", text="from dense", dense_score=0.9)
    second = make_result("x", text="from bm25")

    fused = reciprocal_rank_fusion([[first], [second]])

    assert fused[0].text == "from dense"
    assert fused[0].dense_score == 0.9


def test_confidence_is_zero_without_a_chunk():
    assert compute_confidence(None) == 0.0
    assert should_refuse(compute_confidence(None))


def test_confidence_averages_dense_score_and_sigmoid_of_rerank_logit():
    chunk = make_result("1", score=0.0, dense_score=0.8)

    assert compute_confidence(chunk) == pytest.approx((0.8 + 0.5) / 2)


def test_confidence_includes_groundedness_when_given():
    chunk = make_result("1", score=0.0, dense_score=0.8)

    assert compute_confidence(chunk, groundedness_score=0.2) == pytest.approx((0.8 + 0.5 + 0.2) / 3)


def test_strong_retrieval_is_answered_and_weak_is_refused():
    strong = make_result("1", score=5.0, dense_score=0.85)
    weak = make_result("2", score=-8.0, dense_score=0.3)

    assert not should_refuse(compute_confidence(strong))
    assert should_refuse(compute_confidence(weak))
    assert math.isclose(compute_confidence(weak), (0.3 + 1 / (1 + math.exp(8))) / 2)
