"""End-to-end indexing pipeline: ingest -> chunk -> embed -> store."""

from __future__ import annotations

from pathlib import Path

from core.chunking import chunk_pages
from core.ingestion import DocumentLoader
from core.vectorstore import ChromaStore

DEFAULT_CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def build_index(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    strategy: str = "recursive",
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> ChromaStore:
    loader = DocumentLoader()
    pages = loader.load_directory(corpus_dir)
    if not pages:
        raise RuntimeError(f"No PDFs found in {corpus_dir}")
    print(f"Loaded {len(pages)} pages from {corpus_dir}")

    chunks = chunk_pages(pages, strategy=strategy)
    print(f"Produced {len(chunks)} chunks using '{strategy}' strategy")

    store_kwargs = {}
    if persist_dir is not None:
        store_kwargs["persist_dir"] = persist_dir
    if collection_name is not None:
        store_kwargs["collection_name"] = collection_name

    store = ChromaStore(**store_kwargs)
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    store = build_index()
    print("Index build complete:", store.get_stats())
