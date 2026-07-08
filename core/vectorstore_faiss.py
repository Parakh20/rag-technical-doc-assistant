"""FAISS-backed alternative to ChromaStore - same VectorStore interface
(core/vectorstore.py), embedded/local (no server process). Proves the
VectorStore Protocol is a real abstraction, not a single-implementation
interface.

Trade-offs vs ChromaStore, by design (small demo-scale corpus, not a
distributed system):
- IndexFlatIP is exact but O(n) per search - fine at this corpus size.
- No true upsert: re-adding an existing chunk_id is a no-op rather than an
  overwrite, since FAISS flat indexes don't support in-place vector update
  without a rebuild. Delete the persist dir to reindex from scratch.
- Metadata `where` filtering is done client-side after over-fetching,
  since FAISS itself has no metadata store.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from core.chunking import Chunk
from core.embeddings import EmbeddingModel
from core.vectorstore import SearchResult

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "vectorstore_data_faiss"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
OVERFETCH_MULTIPLIER = 5  # for client-side `where` filtering


class FaissStore:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = "technical_docs",
        embedding_model: EmbeddingModel | None = None,
    ):
        self.persist_dir = Path(persist_dir) / collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedding_model or EmbeddingModel()
        self._index_path = self.persist_dir / INDEX_FILENAME
        self._metadata_path = self.persist_dir / METADATA_FILENAME
        self._index: faiss.Index | None = None
        self._metadatas: list[dict] = []
        self._chunk_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if self._index_path.exists() and self._metadata_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            stored = json.loads(self._metadata_path.read_text())
            self._metadatas = stored["metadatas"]
            self._chunk_ids = stored["chunk_ids"]

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        self._metadata_path.write_text(
            json.dumps({"metadatas": self._metadatas, "chunk_ids": self._chunk_ids})
        )

    def add_documents(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        new_chunks = [c for c in chunks if c.chunk_id not in set(self._chunk_ids)]
        if not new_chunks:
            return
        embeddings = self.embedder.embed_documents(
            [c.text for c in new_chunks], show_progress=False
        ).astype("float32")
        if self._index is None:
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._chunk_ids.extend(c.chunk_id for c in new_chunks)
        self._metadatas.extend(
            {
                "text": c.text, "source": c.source, "page": c.page, "section": c.section,
                "jurisdiction": c.jurisdiction, "doc_type": c.doc_type,
                "parent_id": c.parent_id, "parent_text": c.parent_text,
            }
            for c in new_chunks
        )
        self._save()

    def search(self, query: str, k: int = 10, where: dict | None = None) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        fetch_k = min(k * OVERFETCH_MULTIPLIER if where else k, self._index.ntotal)
        query_embedding = self.embedder.embed_query(query).astype("float32").reshape(1, -1)
        scores, indices = self._index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadatas[idx]
            if where and any(meta.get(key) != value for key, value in where.items()):
                continue
            results.append(
                SearchResult(
                    text=meta["text"], source=meta["source"], page=meta["page"],
                    section=meta["section"], chunk_id=self._chunk_ids[idx], score=float(score),
                    jurisdiction=meta.get("jurisdiction", "n/a"), doc_type=meta.get("doc_type", "n/a"),
                    parent_id=meta.get("parent_id", ""), parent_text=meta.get("parent_text", ""),
                    dense_score=float(score),
                )
            )
            if len(results) >= k:
                break
        return results

    def get_all_documents(self) -> list[dict]:
        return [
            {"chunk_id": cid, "text": meta["text"], "metadata": meta}
            for cid, meta in zip(self._chunk_ids, self._metadatas)
        ]

    def get_stats(self) -> dict:
        sources = sorted({m["source"] for m in self._metadatas})
        return {"total_chunks": len(self._chunk_ids), "total_documents": len(sources), "sources": sources}


if __name__ == "__main__":
    import shutil

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader

    smoke_test_dir = Path(__file__).parent.parent / "vectorstore_data_faiss" / "_smoke_test"
    shutil.rmtree(smoke_test_dir.parent, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:10]
    chunks = chunk_pages(pages, strategy="recursive")

    store = FaissStore(persist_dir=smoke_test_dir, collection_name="smoke_test")
    store.add_documents(chunks)
    print(f"Stats: {store.get_stats()}")

    results = store.search("visual line of sight requirements", k=3)
    for r in results:
        print(f"  score={r.score:.3f} source={r.source} page={r.page} text={r.text[:80]!r}")

    shutil.rmtree(smoke_test_dir.parent, ignore_errors=True)
