"""Shared fakes so the suite runs offline: no model downloads, no API keys.

FakeEmbedder / FakeCrossEncoder stand in for sentence-transformers models
with deterministic bag-of-words scoring, which is enough to exercise the
real vector stores, retrieval, fusion and pipeline wiring end to end.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from core.chunking import Chunk
from core.ingestion import Page
from core.vectorstore import SearchResult

FAKE_DIM = 64
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class FakeEmbedder:
    """Hashed bag-of-words embedding, L2-normalized like the real model."""

    model_name = "fake-embedder"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(FAKE_DIM, dtype="float32")
        for word in _words(text):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % FAKE_DIM
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, FAKE_DIM), dtype="float32")
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    @property
    def dimension(self) -> int:
        return FAKE_DIM


class FakeCrossEncoder:
    """Scores a (query, passage) pair by word overlap, on a logit-like scale."""

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            overlap = len(set(_words(query)) & set(_words(passage)))
            scores.append(float(overlap) * 2.0 - 4.0)
        return np.array(scores)


SAMPLE_PAGES = [
    Page(
        text="Small unmanned aircraft must not fly higher than 400 feet above ground level.",
        source="faa.pdf", page=1, section="Altitude", word_count=12,
        jurisdiction="us", doc_type="regulation",
    ),
    Page(
        text="The remote pilot must keep the aircraft within visual line of sight at all times.",
        source="faa.pdf", page=2, section="VLOS", word_count=15,
        jurisdiction="us", doc_type="regulation",
    ),
    Page(
        text="Every drone in India requires a Unique Identification Number issued by DGCA.",
        source="dgca.pdf", page=1, section="UIN", word_count=12,
        jurisdiction="india", doc_type="regulation",
    ),
]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_cross_encoder() -> FakeCrossEncoder:
    return FakeCrossEncoder()


@pytest.fixture
def sample_pages() -> list[Page]:
    return list(SAMPLE_PAGES)


@pytest.fixture
def sample_chunks(sample_pages) -> list[Chunk]:
    from core.chunking import chunk_pages

    return chunk_pages(sample_pages, strategy="recursive")


@pytest.fixture
def chroma_store(tmp_path, fake_embedder, sample_chunks):
    from core.vectorstore import ChromaStore

    store = ChromaStore(
        persist_dir=tmp_path / "chroma", collection_name="test_docs",
        embedding_model=fake_embedder,
    )
    store.add_documents(sample_chunks)
    return store


def make_result(chunk_id: str, score: float = 0.5, **overrides) -> SearchResult:
    fields = {
        "text": f"text {chunk_id}", "source": "doc.pdf", "page": 1, "section": "",
        "chunk_id": chunk_id, "score": score,
    }
    fields.update(overrides)
    return SearchResult(**fields)
