"""Sentence-level citation verification: for each cited sentence in a
generated answer, check the cited chunk actually supports it, using the
same cross-encoder already loaded for reranking (score, not the strict
entailment interpretation of an NLI model - approved trade-off, see
docs/superpowers/specs/2026-07-08-advanced-rag-upgrade-design.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from core.chunking import split_sentences
from core.vectorstore import SearchResult

CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]")
DEFAULT_SUPPORT_THRESHOLD = -3.0  # same raw cross-encoder logit scale as retrieval.py


@dataclass
class SentenceVerification:
    sentence: str
    cited_source: str | None
    cited_page: int | None
    supported: bool | None  # None = sentence had no citation to check
    score: float | None

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "cited_source": self.cited_source,
            "cited_page": self.cited_page,
            "supported": self.supported,
            "score": self.score,
        }


def _find_cited_chunk(
    source: str, page: int, chunks: list[SearchResult]
) -> SearchResult | None:
    for c in chunks:
        if c.source == source and c.page == page:
            return c
    return None


def verify_citations(
    answer: str,
    chunks: list[SearchResult],
    cross_encoder: CrossEncoder,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> list[SentenceVerification]:
    """Degrades gracefully: a sentence citing a source not present in
    `chunks` is reported unsupported rather than raising."""
    verifications: list[SentenceVerification] = []
    uncited: list[tuple[int, str]] = []
    pairs: list[tuple[str, str]] = []
    pair_targets: list[int] = []

    for sentence in split_sentences(answer):
        match = CITATION_RE.search(sentence)
        if not match:
            verifications.append(
                SentenceVerification(sentence, None, None, None, None)
            )
            continue
        source, page = match.group(1).strip(), int(match.group(2))
        chunk = _find_cited_chunk(source, page, chunks)
        idx = len(verifications)
        verifications.append(SentenceVerification(sentence, source, page, False, None))
        if chunk is not None:
            clean_sentence = CITATION_RE.sub("", sentence).strip()
            pairs.append((clean_sentence, chunk.text))
            pair_targets.append(idx)

    if pairs:
        scores = cross_encoder.predict(pairs)
        for idx, score in zip(pair_targets, scores):
            v = verifications[idx]
            verifications[idx] = SentenceVerification(
                v.sentence, v.cited_source, v.cited_page,
                bool(float(score) >= threshold), float(score),
            )
    return verifications


if __name__ == "__main__":
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunks = [
        SearchResult(
            text="Remote pilots must keep the small UAS within visual line "
                 "of sight at all times during flight operations.",
            source="faa_ac_107-2a_small_uas.pdf", page=12, section="VLOS",
            chunk_id="x1", score=0.9,
        ),
    ]
    answer = (
        "Operators must maintain visual line of sight with their aircraft "
        "[Source: faa_ac_107-2a_small_uas.pdf, Page 12]. "
        "Drones can fly to the moon [Source: faa_ac_107-2a_small_uas.pdf, Page 12]."
    )
    for v in verify_citations(answer, chunks, ce):
        print(v.to_dict())
