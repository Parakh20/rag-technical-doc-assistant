from types import SimpleNamespace

import pytest

import core.generation as generation
from core.generation import RAGGenerator
from tests.conftest import make_result


class _FakeModels:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_token_count=40, candidates_token_count=10)
        return SimpleNamespace(text=self.text, usage_metadata=usage)


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.setattr(generation, "MIN_SECONDS_BETWEEN_CALLS", 0)
    gen = RAGGenerator(api_key="test-key-not-real")
    gen.client = SimpleNamespace(models=_FakeModels("Stay below 400 feet [Source: faa.pdf, Page 1]."))
    return gen


def test_generator_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        RAGGenerator()


def test_user_prompt_includes_numbered_chunks_with_sources():
    prompt = RAGGenerator._build_user_prompt("What is VLOS?", [make_result("1", source="faa.pdf", page=12)])

    assert "[CHUNK 1] (Source: faa.pdf, Page 12)" in prompt
    assert "Question: What is VLOS?" in prompt


def test_build_response_marks_refusals_as_not_grounded():
    chunks = [make_result("1")]

    assert RAGGenerator._build_response("Answer [Source: doc.pdf, Page 1]", chunks, 5).grounded
    assert not RAGGenerator._build_response("The context does not contain this.", chunks, 5).grounded


def test_retry_delay_parses_api_hint_and_caps_it():
    assert generation._extract_retry_delay(Exception("Please retry in 4.5s")) == 5.5
    assert generation._extract_retry_delay(Exception("retry in 300s")) == 35.0
    assert generation._extract_retry_delay(Exception("no hint")) == 15.0


def test_generate_calls_model_and_builds_response(generator):
    chunks = [make_result("1", source="faa.pdf", page=1, score=0.9)]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    response = generator.generate("Max altitude?", chunks, conversation_history=history)

    call = generator.client.models.calls[0]
    assert call["model"] == generation.MODEL_NAME
    assert [c["role"] for c in call["contents"]] == ["user", "model", "user"]
    assert response.tokens_used == 50
    assert response.grounded
    assert response.sources[0].filename == "faa.pdf"
