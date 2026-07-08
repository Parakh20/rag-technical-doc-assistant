"""Grounded generation over retrieved chunks via the Gemini API.

Primary model is the best available Gemini model; if it's overloaded
(503) or quota-exhausted (429) on a given key, falls back to a smaller
model that's confirmed to have working quota - mirrors the BGE -> MiniLM
fallback pattern already used in core/embeddings.py.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from core.metrics import QueryMetrics
from core.vectorstore import SearchResult

load_dotenv(override=True)  # .env wins over a stale key already in the shell

MODEL_NAME = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
MAX_TOKENS = 1024

# The free tier caps gemini-3.1-flash-lite at 15 requests/minute. Pacing
# every API call (generation + groundedness judge share this limiter) at
# this interval keeps a sustained eval run under that cap with margin,
# instead of bursting and getting 429s from q010 onward.
MIN_SECONDS_BETWEEN_CALLS = 6.5
MAX_RATE_LIMIT_RETRIES = 1

_rate_limit_lock = threading.Lock()
_last_call_time = 0.0


def _wait_for_rate_limit() -> None:
    global _last_call_time
    with _rate_limit_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        _last_call_time = time.monotonic()


def _extract_retry_delay(exc: Exception, default: float = 15.0, cap: float = 35.0) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    delay = float(match.group(1)) + 1.0 if match else default
    return min(delay, cap)


def _call_with_rate_limit_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), pacing under the free-tier RPM cap and
    retrying on 429 using the API's own suggested retry delay."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _wait_for_rate_limit()
        try:
            return fn(*args, **kwargs)
        except genai_errors.ClientError as exc:
            if "429" not in str(exc) or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            time.sleep(_extract_retry_delay(exc))

SYSTEM_PROMPT = (
    "You are a precise technical assistant answering questions about "
    "drone regulations and UAS systems. Answer ONLY based on the "
    "provided context. If the context does not contain enough "
    "information to answer fully, say so explicitly. "
    "Always cite your sources using [Source: filename, Page X] format."
)

# Phrases that indicate the model is explicitly declining to answer because
# the context was insufficient - used to flag grounded=False.
REFUSAL_MARKERS = (
    "not found in the context",
    "does not contain",
    "doesn't contain",
    "cannot answer",
    "can't answer",
    "not contain enough information",
    "not available in the provided context",
    "no information",
)


@dataclass
class SourceRef:
    filename: str
    page: int
    section: str
    relevance_score: float

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "relevance_score": self.relevance_score,
        }


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceRef]
    chunks_used: int
    tokens_used: int
    grounded: bool
    confidence: float | None = None
    refused: bool = False
    citations: list[dict] | None = None
    metrics: "QueryMetrics | None" = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "chunks_used": self.chunks_used,
            "tokens_used": self.tokens_used,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "refused": self.refused,
            "citations": self.citations,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class RAGGenerator:
    def __init__(self, api_key: str | None = None, model: str = MODEL_NAME):
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Add it to .env "
                "(see .env.example) or pass api_key explicitly."
            )
        # Explicit timeout (ms): without it, a call to an overloaded model
        # can hang indefinitely instead of raising a fast, retriable error.
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=30_000)
        )
        self.model = model
        self.fallback_model = FALLBACK_MODEL

    @staticmethod
    def _build_user_prompt(query: str, chunks: list[SearchResult]) -> str:
        context = "\n\n".join(
            f"[CHUNK {i}] (Source: {c.source}, Page {c.page})\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer based only on the context above. Cite sources inline."
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ) -> RAGResponse:
        """Non-streaming generation, used by the evaluation suite."""
        contents = self._build_contents(query, retrieved_chunks, conversation_history)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_TOKENS
        )

        try:
            response = _call_with_rate_limit_retry(
                self.client.models.generate_content,
                model=self.model, contents=contents, config=config,
            )
        except genai_errors.APIError:
            response = _call_with_rate_limit_retry(
                self.client.models.generate_content,
                model=self.fallback_model, contents=contents, config=config,
            )

        answer = response.text or ""
        tokens_used = self._extract_tokens(response)
        return self._build_response(answer, retrieved_chunks, tokens_used)

    def stream_generate(
        self,
        query: str,
        retrieved_chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ):
        """Streaming generation for the UI. Yields text deltas; sets
        self.last_response to the final RAGResponse once the generator
        is exhausted."""
        contents = self._build_contents(query, retrieved_chunks, conversation_history)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_TOKENS
        )

        try:
            chunks_iter = _call_with_rate_limit_retry(
                lambda: list(self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                ))
            )
        except genai_errors.APIError:
            chunks_iter = _call_with_rate_limit_retry(
                lambda: list(self.client.models.generate_content_stream(
                    model=self.fallback_model, contents=contents, config=config
                ))
            )

        full_text = ""
        last_chunk = None
        for chunk in chunks_iter:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text
            last_chunk = chunk

        tokens_used = self._extract_tokens(last_chunk) if last_chunk else 0
        self.last_response = self._build_response(full_text, retrieved_chunks, tokens_used)

    @staticmethod
    def _extract_tokens(response) -> int:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0
        return (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

    def _build_contents(
        self,
        query: str,
        retrieved_chunks: list[SearchResult],
        conversation_history: list[dict] | None,
    ) -> list[dict]:
        user_prompt = self._build_user_prompt(query, retrieved_chunks)
        contents = [
            {
                "role": "model" if turn["role"] == "assistant" else "user",
                "parts": [{"text": turn["content"]}],
            }
            for turn in (conversation_history or [])
        ]
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        return contents

    @staticmethod
    def _build_response(
        answer: str, chunks: list[SearchResult], tokens_used: int
    ) -> RAGResponse:
        grounded = not any(marker in answer.lower() for marker in REFUSAL_MARKERS)
        sources = [
            SourceRef(filename=c.source, page=c.page, section=c.section, relevance_score=c.score)
            for c in chunks
        ]
        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            tokens_used=tokens_used,
            grounded=grounded,
        )


if __name__ == "__main__":
    from core.vectorstore import SearchResult as SR

    fake_chunks = [
        SR(text="Remote pilots must keep the small UAS within VLOS at all times.",
           source="faa_ac_107-2a_small_uas.pdf", page=12, section="VLOS",
           chunk_id="x1", score=0.91),
    ]

    print("Built system prompt:", SYSTEM_PROMPT[:60], "...")
    print("Built user prompt:\n", RAGGenerator._build_user_prompt(
        "What is VLOS?", fake_chunks
    ))

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        generator = RAGGenerator()
        response = generator.generate("What is VLOS?", fake_chunks)
        print("\nLive API response:")
        print(response.to_dict())
    else:
        print("\n[skip] GOOGLE_API_KEY not set - prompt construction "
              "verified, skipping live API call.")
