"""Embedding model wrapper around sentence-transformers.

Primary model: BAAI/bge-small-en-v1.5 (384-dim, strong price/performance on
technical text, fast on CPU). Falls back to all-MiniLM-L6-v2 if BGE can't be
loaded (e.g. no network access to download weights on first run).
"""

from __future__ import annotations

import os

# Force offline mode before importing sentence_transformers: huggingface_hub
# otherwise makes a "check for newer files" network request even when the
# model is already cached, and that request hanging is what was actually
# causing multi-minute stalls (not the Claude API calls).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "all-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BATCH_SIZE = 32


class EmbeddingModel:
    def __init__(self, model_name: str = PRIMARY_MODEL, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.model_name, self._model = self._load_with_fallback(model_name)
        self._is_bge = "bge" in self.model_name.lower()

    @staticmethod
    def _load_with_fallback(model_name: str) -> tuple[str, SentenceTransformer]:
        try:
            return model_name, SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - any load failure triggers fallback
            if model_name == FALLBACK_MODEL:
                raise
            print(f"[EmbeddingModel] Failed to load {model_name} ({exc}); "
                  f"falling back to {FALLBACK_MODEL}")
            return FALLBACK_MODEL, SentenceTransformer(FALLBACK_MODEL)

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension))
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{BGE_QUERY_PREFIX}{query}" if self._is_bge else query
        return self._model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        )

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    model = EmbeddingModel()
    print(f"Loaded model: {model.model_name}, dimension={model.dimension}")

    docs = [
        "Remote pilots must maintain visual line of sight with the aircraft.",
        "The maximum altitude for small UAS operations is 400 feet AGL.",
        "RPAS operators in India require a Unique Identification Number.",
    ]
    doc_embeddings = model.embed_documents(docs, show_progress=False)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    query_embedding = model.embed_query("What is the maximum drone altitude?")
    print(f"Query embedding shape: {query_embedding.shape}")

    similarities = doc_embeddings @ query_embedding
    best_idx = int(np.argmax(similarities))
    print(f"Most similar doc (score={similarities[best_idx]:.3f}): {docs[best_idx]!r}")
