"""Qdrant-backed alternative to ChromaStore - same VectorStore interface
(core/vectorstore.py), running in Qdrant's embedded/local mode (on-disk,
no server process, no Docker) to keep the project self-contained.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models

from core.chunking import Chunk
from core.embeddings import EmbeddingModel
from core.vectorstore import SearchResult

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "vectorstore_data_qdrant"


class QdrantStore:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = "technical_docs",
        embedding_model: EmbeddingModel | None = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedding_model or EmbeddingModel()
        self.client = QdrantClient(path=str(self.persist_dir))
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension, distance=models.Distance.COSINE
                ),
            )

    def add_documents(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed_documents(
                [c.text for c in batch], show_progress=False
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=c.chunk_id,
                        vector=embeddings[j].tolist(),
                        payload={
                            "text": c.text, "source": c.source, "page": c.page,
                            "section": c.section, "jurisdiction": c.jurisdiction,
                            "doc_type": c.doc_type, "parent_id": c.parent_id,
                            "parent_text": c.parent_text,
                        },
                    )
                    for j, c in enumerate(batch)
                ],
            )

    def search(self, query: str, k: int = 10, where: dict | None = None) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        query_filter = None
        if where:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in where.items()
                ]
            )
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=k,
        ).points
        return [
            SearchResult(
                text=hit.payload["text"], source=hit.payload["source"], page=hit.payload["page"],
                section=hit.payload["section"], chunk_id=str(hit.id), score=hit.score,
                jurisdiction=hit.payload.get("jurisdiction", "n/a"),
                doc_type=hit.payload.get("doc_type", "n/a"),
                parent_id=hit.payload.get("parent_id", ""), parent_text=hit.payload.get("parent_text", ""),
                dense_score=hit.score,
            )
            for hit in hits
        ]

    def get_all_documents(self) -> list[dict]:
        total = self.client.count(self.collection_name).count
        if total == 0:
            return []
        points, _ = self.client.scroll(self.collection_name, limit=total)
        return [
            {"chunk_id": str(p.id), "text": p.payload["text"], "metadata": p.payload}
            for p in points
        ]

    def get_stats(self) -> dict:
        total = self.client.count(self.collection_name).count
        if total == 0:
            return {"total_chunks": 0, "total_documents": 0, "sources": []}
        points, _ = self.client.scroll(self.collection_name, limit=total)
        sources = sorted({p.payload["source"] for p in points})
        return {"total_chunks": total, "total_documents": len(sources), "sources": sources}


if __name__ == "__main__":
    import shutil

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader

    smoke_test_dir = Path(__file__).parent.parent / "vectorstore_data_qdrant" / "_smoke_test"
    shutil.rmtree(smoke_test_dir.parent, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:10]
    chunks = chunk_pages(pages, strategy="recursive")

    store = QdrantStore(persist_dir=smoke_test_dir, collection_name="smoke_test")
    store.add_documents(chunks)
    print(f"Stats: {store.get_stats()}")

    results = store.search("visual line of sight requirements", k=3)
    for r in results:
        print(f"  score={r.score:.3f} source={r.source} page={r.page} text={r.text[:80]!r}")

    shutil.rmtree(smoke_test_dir.parent, ignore_errors=True)
