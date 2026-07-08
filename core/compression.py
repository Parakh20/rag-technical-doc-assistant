"""Extractive context compression: shrink each retrieved chunk down to its
most query-relevant sentences before it hits the generator, cutting tokens
without an extra LLM call (no added latency/cost, unlike an LLM-based
summarizer - see docs/superpowers/specs/2026-07-08-advanced-rag-upgrade-design.md).
"""

from __future__ import annotations

import numpy as np

from core.chunking import count_tokens, split_sentences
from core.embeddings import EmbeddingModel
from core.vectorstore import SearchResult

DEFAULT_TOKEN_BUDGET_PER_CHUNK = 150


def compress_chunk(
    query_embedding: np.ndarray,
    text: str,
    embedder: EmbeddingModel,
    token_budget: int = DEFAULT_TOKEN_BUDGET_PER_CHUNK,
) -> str:
    """Keep the highest-scoring sentences (by cosine similarity to the query
    embedding) up to token_budget, in their original order."""
    sentences = split_sentences(text)
    if len(sentences) <= 1 or count_tokens(text) <= token_budget:
        return text

    sentence_embeddings = embedder.embed_documents(sentences, show_progress=False)
    scores = sentence_embeddings @ query_embedding  # pre-normalized embeddings

    kept_indices: list[int] = []
    budget_used = 0
    for idx in sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True):
        cost = count_tokens(sentences[idx])
        if kept_indices and budget_used + cost > token_budget:
            continue
        kept_indices.append(idx)
        budget_used += cost

    if not kept_indices:
        return text
    return " ".join(sentences[i] for i in sorted(kept_indices))


def compress_chunks(
    query: str,
    chunks: list[SearchResult],
    embedder: EmbeddingModel,
    token_budget: int = DEFAULT_TOKEN_BUDGET_PER_CHUNK,
) -> list[SearchResult]:
    if not chunks:
        return []
    query_embedding = embedder.embed_query(query)
    return [
        SearchResult(
            text=compress_chunk(query_embedding, c.text, embedder, token_budget),
            source=c.source, page=c.page, section=c.section, chunk_id=c.chunk_id,
            score=c.score, jurisdiction=c.jurisdiction, doc_type=c.doc_type,
            parent_id=c.parent_id, parent_text=c.parent_text,
        )
        for c in chunks
    ]


if __name__ == "__main__":
    from core.embeddings import EmbeddingModel as EM

    embedder = EM()
    sample = SearchResult(
        text=(
            "Remote pilots must keep the small UAS within visual line of sight "
            "at all times during flight operations. The aircraft must not "
            "exceed an altitude of 400 feet above ground level unless within "
            "a 400-foot radius of a structure. Weather conditions must meet "
            "minimum visibility requirements of three statute miles. The "
            "operator must hold a valid remote pilot certificate issued under "
            "Part 107 before conducting any commercial operations."
        ),
        source="faa_ac_107-2a_small_uas.pdf", page=12, section="VLOS",
        chunk_id="x1", score=0.9,
    )
    compressed = compress_chunks(
        "What is the maximum altitude for small UAS?", [sample], embedder, token_budget=20
    )
    print("Original tokens:", count_tokens(sample.text))
    print("Compressed tokens:", count_tokens(compressed[0].text))
    print("Compressed text:", compressed[0].text)
