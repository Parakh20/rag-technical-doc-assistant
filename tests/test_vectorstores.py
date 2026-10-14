import pytest

from core.vectorstore import ChromaStore, VectorStore
from core.vectorstore_faiss import FaissStore
from core.vectorstore_qdrant import QdrantStore


@pytest.fixture(params=["chroma", "faiss", "qdrant"])
def store(request, tmp_path, fake_embedder, sample_chunks):
    cls = {"chroma": ChromaStore, "faiss": FaissStore, "qdrant": QdrantStore}[request.param]
    instance = cls(persist_dir=tmp_path / request.param, collection_name="test_docs",
                   embedding_model=fake_embedder)
    instance.add_documents(sample_chunks)
    yield instance
    client = getattr(instance, "client", None)
    if request.param == "qdrant" and client is not None:
        client.close()


def test_store_conforms_to_vector_store_protocol(store):
    assert isinstance(store, VectorStore)


def test_search_returns_most_similar_chunk_first(store):
    results = store.search("drone unique identification number india", k=3)

    assert results[0].source == "dgca.pdf"
    assert results[0].jurisdiction == "india"
    assert len(results) == 3


def test_search_applies_metadata_filter(store):
    results = store.search("aircraft", k=5, where={"jurisdiction": "us"})

    assert results
    assert {r.jurisdiction for r in results} == {"us"}


def test_get_stats_and_all_documents(store, sample_chunks):
    stats = store.get_stats()
    docs = store.get_all_documents()

    assert stats["total_chunks"] == len(sample_chunks)
    assert sorted(stats["sources"]) == ["dgca.pdf", "faa.pdf"]
    assert {d["chunk_id"] for d in docs} == {c.chunk_id for c in sample_chunks}


def test_re_adding_same_chunks_is_idempotent(store, sample_chunks):
    store.add_documents(sample_chunks)

    assert store.get_stats()["total_chunks"] == len(sample_chunks)


def test_faiss_store_persists_and_reloads(tmp_path, fake_embedder, sample_chunks):
    FaissStore(persist_dir=tmp_path, embedding_model=fake_embedder).add_documents(sample_chunks)

    reloaded = FaissStore(persist_dir=tmp_path, embedding_model=fake_embedder)

    assert reloaded.get_stats()["total_chunks"] == len(sample_chunks)
