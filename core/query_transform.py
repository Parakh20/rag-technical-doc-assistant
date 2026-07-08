"""LLM-backed query transforms: multi-query expansion and HyDE.

Both are opt-in (each adds one Gemini call per query on top of generation)
and both degrade gracefully to a no-op on API failure rather than raising -
retrieval should never fail just because a reformulation call errored.
"""

from __future__ import annotations

from google.genai import errors as genai_errors

from core.generation import RAGGenerator, _call_with_rate_limit_retry
from google.genai import types

DEFAULT_NUM_EXPANSIONS = 3

EXPANSION_PROMPT = (
    "Generate {n} alternative phrasings of the following question, one per "
    "line, no numbering or extra commentary. Include likely synonyms, "
    "abbreviations, or technical terms a regulatory document might use.\n\n"
    "Question: {query}"
)

HYDE_PROMPT = (
    "Write a short (2-4 sentence) hypothetical answer to the following "
    "question, as if it were an excerpt from a technical regulatory "
    "document. This will be used purely for retrieval, not shown to a "
    "user - do not hedge or say you don't know.\n\n"
    "Question: {query}"
)


def _generate_text(generator: RAGGenerator, prompt: str, max_tokens: int) -> str:
    config = types.GenerateContentConfig(max_output_tokens=max_tokens)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    try:
        response = _call_with_rate_limit_retry(
            generator.client.models.generate_content,
            model=generator.model, contents=contents, config=config,
        )
    except genai_errors.APIError:
        response = _call_with_rate_limit_retry(
            generator.client.models.generate_content,
            model=generator.fallback_model, contents=contents, config=config,
        )
    return (response.text or "").strip()


def expand_query(
    generator: RAGGenerator, query: str, num_expansions: int = DEFAULT_NUM_EXPANSIONS
) -> list[str]:
    """Return up to num_expansions alternative phrasings (excluding the
    original query). Empty list on any failure - caller falls back to
    single-query retrieval."""
    try:
        text = _generate_text(
            generator, EXPANSION_PROMPT.format(n=num_expansions, query=query), max_tokens=200
        )
    except Exception:  # noqa: BLE001 - never let expansion failure break retrieval
        return []
    variants = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
    return variants[:num_expansions]


def hyde_hypothetical_answer(generator: RAGGenerator, query: str) -> str | None:
    """Return a hypothetical-document string to embed instead of/alongside
    the raw query, or None on failure."""
    try:
        return _generate_text(generator, HYDE_PROMPT.format(query=query), max_tokens=150)
    except Exception:  # noqa: BLE001 - never let HyDE failure break retrieval
        return None


if __name__ == "__main__":
    import os

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        gen = RAGGenerator()
        query = "What is the maximum altitude for small UAS operations?"
        print("Expansions:", expand_query(gen, query))
        print("HyDE:", hyde_hypothetical_answer(gen, query))
    else:
        print("[skip] GOOGLE_API_KEY not set - skipping live API call.")
