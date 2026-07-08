"""In-memory BM25 lexical retrieval, complementing dense embeddings.

Dense retrieval compresses semantics into a fixed-size vector and can miss
exact-match signals - regulation IDs, abbreviations (UIN, VLOS), numbers
(400 feet). BM25 is a plain term-frequency ranker over the same chunk
corpus and catches those. Combined with dense search via reciprocal rank
fusion (core/fusion.py), this is "hybrid retrieval".
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from core.vectorstore import ChromaStore, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Built once from a store's full corpus; independent of the dense index."""

    def __init__(self, store: ChromaStore):
        documents = store.get_all_documents()
        self._ids = [d["chunk_id"] for d in documents]
        self._texts = [d["text"] for d in documents]
        self._metadatas = [d["metadata"] for d in documents]
        corpus_tokens = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, k: int = 20) -> list[SearchResult]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in ranked_indices:
            if scores[i] <= 0:
                continue
            meta = self._metadatas[i]
            results.append(
                SearchResult(
                    text=self._texts[i],
                    source=meta.get("source", ""),
                    page=meta.get("page", 0),
                    section=meta.get("section", ""),
                    chunk_id=self._ids[i],
                    score=float(scores[i]),
                    jurisdiction=meta.get("jurisdiction", "n/a"),
                    doc_type=meta.get("doc_type", "n/a"),
                    parent_id=meta.get("parent_id", ""),
                    parent_text=meta.get("parent_text", ""),
                )
            )
        return results


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader

    smoke_test_dir = Path(__file__).parent.parent / "vectorstore_data" / "_smoke_test_bm25"
    shutil.rmtree(smoke_test_dir, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:30]
    chunks = chunk_pages(pages, strategy="recursive")

    store = ChromaStore(persist_dir=smoke_test_dir, collection_name="smoke_test_bm25")
    store.add_documents(chunks)

    bm25 = BM25Retriever(store)
    for r in bm25.search("visual line of sight VLOS requirements", k=3):
        print(f"  score={r.score:.3f} page={r.page} text={r.text[:80]!r}")

    shutil.rmtree(smoke_test_dir, ignore_errors=True)
