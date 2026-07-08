"""Three chunking strategies, compared empirically in evaluation/metrics.py.

  1. FixedSizeChunker      - fixed token windows with overlap
  2. RecursiveCharacterChunker (default for the main index) - splits on
     paragraph/sentence/word boundaries, target 400-600 tokens
  3. SemanticChunker       - groups sentences by embedding similarity

All three take a list of core.ingestion.Page and emit Chunk objects that
carry the page's metadata forward plus chunk_id/chunk_index/strategy.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import tiktoken

from core.ingestion import Page

TOKEN_ENCODING = "cl100k_base"
_encoder = tiktoken.get_encoding(TOKEN_ENCODING)

# Fixed namespace for deterministic chunk_id generation (uuid5), so that
# re-running the same strategy over the same corpus produces the same IDs
# and ChromaStore.add_documents' upsert is actually idempotent.
_CHUNK_ID_NAMESPACE = uuid.UUID("8f8e6f2a-6c2e-4a3b-9a8e-2f6a7b4c9d1e")


def _make_chunk_id(source: str, strategy: str, chunk_index: int) -> str:
    key = f"{source}:{strategy}:{chunk_index}"
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key))


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    section: str
    word_count: int
    chunk_id: str
    chunk_index: int
    strategy: str
    jurisdiction: str
    doc_type: str
    parent_id: str
    parent_text: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "word_count": self.word_count,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "jurisdiction": self.jurisdiction,
            "doc_type": self.doc_type,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
        }


class FixedSizeChunker:
    """Strategy 1: fixed-size token windows with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_page(self, page: Page) -> list[str]:
        tokens = _encoder.encode(page.text)
        if not tokens:
            return []
        step = max(self.chunk_size - self.overlap, 1)
        chunks = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(_encoder.decode(window))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks


class RecursiveCharacterChunker:
    """Strategy 2 (default): recursively split on paragraph -> line -> sentence
    -> word boundaries, targeting min_tokens-max_tokens per chunk."""

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, min_tokens: int = 400, max_tokens: int = 600):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk_page(self, page: Page) -> list[str]:
        return self._split(page.text, list(self.SEPARATORS))

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        if count_tokens(text) <= self.max_tokens:
            return [text]
        if not separators:
            return self._hard_split(text)

        sep, rest_seps = separators[0], separators[1:]
        parts = [p for p in text.split(sep) if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{sep}{part}" if buffer else part
            if count_tokens(candidate) <= self.max_tokens:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
            if count_tokens(part) > self.max_tokens:
                chunks.extend(self._split(part, rest_seps))
                buffer = ""
            else:
                buffer = part
        if buffer:
            chunks.append(buffer)
        return [c.strip() for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        tokens = _encoder.encode(text)
        return [
            _encoder.decode(tokens[i : i + self.max_tokens])
            for i in range(0, len(tokens), self.max_tokens)
        ]


class SemanticChunker:
    """Strategy 3: split where adjacent-sentence cosine similarity drops
    below a threshold, grouping semantically related sentences together."""

    def __init__(
        self, similarity_threshold: float = 0.7, model_name: str = "all-MiniLM-L6-v2"
    ):
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk_page(self, page: Page) -> list[str]:
        sentences = self._split_sentences(page.text)
        if len(sentences) <= 1:
            return sentences
        model = self._get_model()
        embeddings = model.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True
        )
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
            if similarity >= self.similarity_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])
        return [" ".join(g).strip() for g in groups if " ".join(g).strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]


STRATEGIES = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveCharacterChunker,
    "semantic": SemanticChunker,
}


def chunk_pages(pages: list[Page], strategy: str = "recursive", **kwargs) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. "
                          f"Choose from {list(STRATEGIES)}")
    chunker = STRATEGIES[strategy](**kwargs)
    per_source_index: dict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunker.chunk_page(page):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    text=text,
                    source=page.source,
                    page=page.page,
                    section=page.section,
                    word_count=len(text.split()),
                    chunk_id=_make_chunk_id(page.source, strategy, per_source_index[page.source]),
                    chunk_index=per_source_index[page.source],
                    strategy=strategy,
                    jurisdiction=page.jurisdiction,
                    doc_type=page.doc_type,
                    parent_id=f"{page.source}:p{page.page}",
                    parent_text=page.text,
                )
            )
            per_source_index[page.source] += 1
    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from core.ingestion import DocumentLoader

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)
    print(f"Loaded {len(pages)} pages")

    for strategy in ("fixed", "recursive"):
        chunks = chunk_pages(pages, strategy=strategy)
        sizes = [count_tokens(c.text) for c in chunks]
        print(f"[{strategy}] {len(chunks)} chunks, "
              f"avg {sum(sizes)/len(sizes):.0f} tokens, "
              f"min {min(sizes)}, max {max(sizes)}")

    # Semantic chunking is slower (loads a sentence-transformer model);
    # smoke test it on a small subset only.
    sample_pages = pages[:3]
    semantic_chunks = chunk_pages(sample_pages, strategy="semantic")
    print(f"[semantic] {len(semantic_chunks)} chunks on {len(sample_pages)} sample pages")
    if semantic_chunks:
        print("Sample chunk:", semantic_chunks[0].to_dict())
