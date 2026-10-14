import json

import pytest
from fastapi.testclient import TestClient

from core.generation import RAGGenerator
from pipeline.query import QueryPipeline


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.last_response = None

    def generate(self, query, chunks, conversation_history=None):
        self.calls += 1
        answer = f"Answer about {query} [Source: {chunks[0].source}, Page {chunks[0].page}]."
        self.last_response = RAGGenerator._build_response(answer, chunks, tokens_used=42)
        return self.last_response

    def stream_generate(self, query, chunks, conversation_history=None):
        yield "Answer "
        yield "streamed."
        self.last_response = RAGGenerator._build_response("Answer streamed.", chunks, tokens_used=7)


@pytest.fixture
def pipeline(chroma_store, fake_cross_encoder):
    p = QueryPipeline(store=chroma_store, generator=FakeGenerator())
    p.retriever._cross_encoder = fake_cross_encoder
    return p


def test_pipeline_answers_with_sources_confidence_and_metrics(pipeline):
    response = pipeline.ask("How high can a small unmanned aircraft fly?", final_k=2)

    assert response.sources[0].filename == "faa.pdf"
    assert response.confidence is not None
    assert response.metrics.num_chunks_retrieved == 2
    assert response.metrics.tokens_used == 42


def test_pipeline_caches_identical_stateless_queries(pipeline):
    first = pipeline.ask("What is VLOS?")
    second = pipeline.ask("  what is vlos? ")

    assert pipeline.generator.calls == 1
    assert second.metrics.cache_hit and not first.metrics.cache_hit


def test_pipeline_skips_cache_when_conversation_history_given(pipeline):
    history = [{"role": "user", "content": "hi"}]
    pipeline.ask("What is VLOS?", conversation_history=history)
    pipeline.ask("What is VLOS?", conversation_history=history)

    assert pipeline.generator.calls == 2


def test_pipeline_verifies_citations_when_asked(pipeline):
    response = pipeline.ask("drone identification number india", do_verify_citations=True)

    assert response.citations and response.citations[0]["cited_source"] == "dgca.pdf"


def test_raptor_requires_tree_path(pipeline):
    with pytest.raises(ValueError, match="raptor_tree_path"):
        pipeline.ask("q", use_raptor=True)


@pytest.fixture
def client(monkeypatch, pipeline):
    import api.main

    monkeypatch.setattr(api.main, "QueryPipeline", lambda: pipeline)
    with TestClient(api.main.app) as test_client:
        yield test_client


def test_health_reports_corpus_stats(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["corpus"]["total_documents"] == 2


def test_sync_query_returns_validated_response(client):
    resp = client.post("/query/sync", json={"query": "visual line of sight"})

    assert resp.status_code == 200
    assert resp.json()["sources"]


def test_stream_query_emits_sse_deltas_then_final(client):
    resp = client.post("/query", json={"query": "visual line of sight"})
    events = [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]

    assert [e["delta"] for e in events if "delta" in e] == ["Answer ", "streamed."]
    assert events[-1]["done"] is True
    assert events[-1]["final"]["answer"] == "Answer streamed."
