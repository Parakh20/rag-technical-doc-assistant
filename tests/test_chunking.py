import pytest

from core.chunking import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    chunk_pages,
    count_tokens,
    split_sentences,
)
from core.ingestion import Page


def _page(text: str, source: str = "doc.pdf", page: int = 1) -> Page:
    return Page(text=text, source=source, page=page, section="S", word_count=len(text.split()),
                jurisdiction="us", doc_type="regulation")


def test_split_sentences_splits_on_terminal_punctuation():
    assert split_sentences("One. Two!  Three? ") == ["One.", "Two!", "Three?"]


def test_fixed_size_chunker_respects_window_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = FixedSizeChunker(chunk_size=100, overlap=10).chunk_page(_page(text))

    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_recursive_chunker_keeps_short_text_whole():
    assert RecursiveCharacterChunker().chunk_page(_page("Short paragraph.")) == ["Short paragraph."]


def test_recursive_chunker_splits_long_text_under_max_tokens():
    paragraphs = "\n\n".join(" ".join(["altitude"] * 120) for _ in range(6))
    chunker = RecursiveCharacterChunker(min_tokens=50, max_tokens=200)

    chunks = chunker.chunk_page(_page(paragraphs))

    assert len(chunks) > 1
    assert all(count_tokens(c) <= 200 for c in chunks)


def test_chunk_pages_ids_are_deterministic_and_carry_metadata(sample_pages):
    first = chunk_pages(sample_pages, strategy="recursive")
    second = chunk_pages(sample_pages, strategy="recursive")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].parent_id == "faa.pdf:p1"
    assert first[0].jurisdiction == "us"
    assert [c.chunk_index for c in first if c.source == "faa.pdf"] == [0, 1]


def test_chunk_ids_differ_between_strategies(sample_pages):
    fixed = chunk_pages(sample_pages, strategy="fixed")
    recursive = chunk_pages(sample_pages, strategy="recursive")

    assert {c.chunk_id for c in fixed}.isdisjoint({c.chunk_id for c in recursive})


def test_chunk_pages_rejects_unknown_strategy(sample_pages):
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_pages(sample_pages, strategy="nope")
