import pytest

from core.bm25 import BM25Retriever
from core.citation_check import verify_citations
from core.compression import compress_chunks
from core.retrieval import RetrieverWithReranker
from tests.conftest import make_result


@pytest.fixture
def retriever(chroma_store, fake_cross_encoder):
    r = RetrieverWithReranker(chroma_store)
    r._cross_encoder = fake_cross_encoder  # skip downloading the real model
    return r


def test_bm25_finds_exact_term_matches(chroma_store):
    results = BM25Retriever(chroma_store).search("Unique Identification Number", k=5)

    assert results[0].source == "dgca.pdf"
    assert all(r.score > 0 for r in results)


def test_bm25_on_empty_store_returns_nothing(tmp_path, fake_embedder):
    from core.vectorstore import ChromaStore

    empty = ChromaStore(persist_dir=tmp_path / "empty", embedding_model=fake_embedder)

    assert BM25Retriever(empty).search("anything") == []


def test_retrieve_without_reranker_returns_dense_top_k(retriever):
    results = retriever.retrieve("400 feet altitude", final_k=2, use_reranker=False)

    assert len(results) == 2
    assert results[0].section == "Altitude"


def test_retrieve_with_reranker_orders_by_cross_encoder_score(retriever):
    results = retriever.retrieve("visual line of sight pilot", final_k=3)

    assert results[0].section == "VLOS"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_hybrid_retrieval_fuses_dense_and_bm25(retriever):
    results = retriever.retrieve("DGCA identification", final_k=3, use_hybrid=True)

    assert results[0].source == "dgca.pdf"


def test_expand_to_parent_swaps_in_full_page_text(retriever):
    results = retriever.retrieve("400 feet", final_k=1, use_reranker=False, expand_to_parent=True)

    assert "400 feet" in results[0].text


def test_mmr_returns_distinct_results(retriever):
    results = retriever.retrieve_mmr("aircraft pilot drone", final_k=3)

    assert len({r.chunk_id for r in results}) == len(results) == 3


def test_compress_chunks_keeps_query_relevant_sentences(fake_embedder):
    chunk = make_result(
        "1",
        text=("The aircraft must stay below 400 feet altitude. "
              "Weather minimums require three miles of visibility. "
              "Operators need a remote pilot certificate."),
    )

    compressed = compress_chunks("maximum altitude 400 feet", [chunk], fake_embedder, token_budget=12)

    assert "400 feet" in compressed[0].text
    assert len(compressed[0].text) < len(chunk.text)


def test_verify_citations_flags_supported_unsupported_and_uncited(fake_cross_encoder):
    chunks = [make_result("1", text="Keep the drone within visual line of sight.",
                          source="faa.pdf", page=2)]
    answer = ("Pilots keep the drone within visual line of sight [Source: faa.pdf, Page 2]. "
              "Night flights need waivers [Source: other.pdf, Page 9]. "
              "This sentence has no citation.")

    results = verify_citations(answer, chunks, fake_cross_encoder)

    assert [r.supported for r in results] == [True, False, None]
    assert results[0].cited_page == 2
