"""FastAPI service wrapping QueryPipeline for concurrent request handling
and SSE-streamed generation.

The underlying libraries (chromadb, sentence-transformers, the Gemini
client) are synchronous - there's no async-native driver for any of them.
"full async" here means what it concretely can: async endpoints that run
the blocking pipeline calls via asyncio.to_thread, so the event loop stays
free to accept/serve other concurrent requests while one query's
retrieval+generation is in flight, and true SSE token streaming to the
client instead of waiting for the full answer.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse
from core.generation import RAGGenerator
from pipeline.query import QueryPipeline

_pipeline: QueryPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = QueryPipeline()
    yield
    _pipeline = None


app = FastAPI(title="RAG Technical Doc Assistant API", lifespan=lifespan)


def get_pipeline() -> QueryPipeline:
    if _pipeline is None:
        raise RuntimeError("QueryPipeline not initialized - app lifespan hasn't started")
    return _pipeline


@app.get("/health")
async def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "corpus": pipeline.store.get_stats()}


@app.post("/query/sync", response_model=QueryResponse)
async def query_sync(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    payload = request.model_dump()
    response = await asyncio.to_thread(pipeline.ask, **payload)
    return QueryResponse.model_validate(response.to_dict())


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/query")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    pipeline = get_pipeline()

    def retrieve_chunks():
        if request.use_mmr:
            return pipeline.retriever.retrieve_mmr(
                request.query, dense_k=request.dense_k, final_k=request.final_k
            )
        if request.use_correction:
            chunks, _iterations = pipeline.retriever.retrieve_with_correction(
                request.query, dense_k=request.dense_k, final_k=request.final_k,
                use_hybrid=request.use_hybrid, where=request.where,
                expand_to_parent=request.expand_to_parent,
            )
            return chunks
        return pipeline.retriever.retrieve(
            request.query, dense_k=request.dense_k, final_k=request.final_k,
            use_reranker=request.use_reranker, use_hybrid=request.use_hybrid,
            use_query_expansion=request.use_query_expansion, use_hyde=request.use_hyde,
            where=request.where, expand_to_parent=request.expand_to_parent,
        )

    chunks = await asyncio.to_thread(retrieve_chunks)
    if request.use_compression:
        from core.compression import compress_chunks

        chunks = await asyncio.to_thread(
            compress_chunks, request.query, chunks, pipeline.retriever.store.embedder
        )

    async def event_stream():
        chunk_queue: queue.Queue = queue.Queue()
        SENTINEL = object()

        def produce():
            generator: RAGGenerator = pipeline.generator
            try:
                for delta in generator.stream_generate(
                    request.query, chunks, conversation_history=request.conversation_history
                ):
                    chunk_queue.put(delta)
            except Exception as exc:  # noqa: BLE001 - forward to the client instead of dropping the stream
                chunk_queue.put(exc)
            finally:
                chunk_queue.put(SENTINEL)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, chunk_queue.get)
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                yield _sse_event({"error": str(item)})
                return
            yield _sse_event({"delta": item})

        final_response = pipeline.generator.last_response
        yield _sse_event({"done": True, "final": final_response.to_dict()})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
