"""Persistent Chroma vector store interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from core.chunking import Chunk
from core.embeddings import EmbeddingModel

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "vectorstore_data"
DEFAULT_COLLECTION_NAME = "technical_docs"
DEFAULT_ADD_BATCH_SIZE = 64


@dataclass
class SearchResult:
    text: str
    source: str
    page: int
    section: str
    chunk_id: str
    score: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


class ChromaStore:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedding_model or EmbeddingModel()

    def add_documents(self, chunks: list[Chunk], batch_size: int = DEFAULT_ADD_BATCH_SIZE) -> None:
        """Embed and upsert chunks (by chunk_id, so re-running is idempotent)."""
        if not chunks:
            return
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing chunks"):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed_documents(texts, show_progress=False)
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "section": c.section,
                        "word_count": c.word_count,
                        "chunk_index": c.chunk_index,
                        "strategy": c.strategy,
                    }
                    for c in batch
                ],
            )

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=k
        )
        if not results["ids"][0]:
            return []
        search_results = []
        for id_, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist  # cosine distance -> similarity
            search_results.append(
                SearchResult(
                    text=doc,
                    source=meta.get("source", ""),
                    page=meta.get("page", 0),
                    section=meta.get("section", ""),
                    chunk_id=id_,
                    score=similarity,
                )
            )
        return search_results

    def get_stats(self) -> dict:
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return {"total_chunks": 0, "total_documents": 0, "sources": []}
        sample = self.collection.get(limit=total_chunks, include=["metadatas"])
        sources = sorted({m.get("source", "") for m in sample["metadatas"]})
        return {
            "total_chunks": total_chunks,
            "total_documents": len(sources),
            "sources": sources,
        }


if __name__ == "__main__":
    import shutil

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader

    smoke_test_dir = Path(__file__).parent.parent / "vectorstore_data" / "_smoke_test"
    shutil.rmtree(smoke_test_dir, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:10]
    chunks = chunk_pages(pages, strategy="recursive")
    print(f"Indexing {len(chunks)} chunks from {len(pages)} sample pages")

    store = ChromaStore(persist_dir=smoke_test_dir, collection_name="smoke_test")
    store.add_documents(chunks)

    stats = store.get_stats()
    print(f"Stats: {stats}")

    results = store.search("visual line of sight requirements", k=3)
    for r in results:
        print(f"  score={r.score:.3f} source={r.source} page={r.page} text={r.text[:80]!r}")

    shutil.rmtree(smoke_test_dir, ignore_errors=True)
