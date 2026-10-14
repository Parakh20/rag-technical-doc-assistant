import json
from collections import Counter

from evaluation.eval_set import EVAL_QUESTIONS, EVAL_SET_PATH, build_eval_set


def test_eval_set_has_documented_composition():
    counts = Counter(q["question_type"] for q in build_eval_set())

    assert counts == {"factual": 25, "procedural": 10, "cross-doc": 5, "unanswerable": 10}


def test_eval_set_ids_are_unique_and_unanswerables_have_no_source():
    assert len({q["id"] for q in EVAL_QUESTIONS}) == len(EVAL_QUESTIONS)
    for q in EVAL_QUESTIONS:
        assert (q["expected_source"] is None) == (q["question_type"] == "unanswerable")


def test_committed_json_matches_python_definition():
    assert json.loads(EVAL_SET_PATH.read_text()) == EVAL_QUESTIONS
