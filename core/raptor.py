"""RAPTOR-style hierarchical tree retrieval: recursively cluster chunks and
summarize each cluster with an LLM, building higher levels of abstraction
on top of the raw chunk corpus. Retrieval then searches across every level
at once (the "collapsed tree" approach from the RAPTOR paper - simpler
than beam-searching the tree top-down, and works fine at this corpus size).

Building the tree is offline/one-time (scripts/build_raptor_tree.py) - each
internal node costs one LLM summarization call, so this is not something to
run on the query path.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import httpx
import numpy as np
from google.genai import errors as genai_errors
from google.genai import types

from core.embeddings import EmbeddingModel
from core.generation import RAGGenerator, _call_with_rate_limit_retry

DEFAULT_CLUSTER_SIZE = 5
DEFAULT_MAX_LEVELS = 3
MIN_NODES_TO_CLUSTER = 3  # below this, stop recursing - nothing meaningful left to summarize
SUMMARY_MAX_TOKENS = 200
# Building the tree means many sequential LLM calls over a long-running batch
# job - transient DNS/connection blips (observed in this environment; see
# README's note on the same issue in evaluation/metrics.py) shouldn't abort
# the whole build. Retried here rather than in core/generation.py's shared
# rate limiter, since that limiter is also used on the live query path where
# a hard fast-fail (not a multi-attempt retry loop) is the right behavior.
MAX_TRANSIENT_RETRIES = 3
TRANSIENT_RETRY_DELAY_SECONDS = 5.0

SUMMARY_PROMPT = (
    "Summarize the key facts across these excerpts from a technical/regulatory "
    "document in one concise paragraph. Preserve specific numbers, thresholds, "
    "and defined terms rather than generalizing them away.\n\n{excerpts}"
)


@dataclass
class RaptorNode:
    node_id: str
    level: int
    text: str
    embedding: list[float]
    child_ids: list[str]
    source: str  # original source filename if a leaf's provenance is knowable, else "summary"


def _cluster_indices(embeddings: np.ndarray, cluster_size: int) -> list[list[int]]:
    n = embeddings.shape[0]
    n_clusters = max(1, round(n / cluster_size))
    if n_clusters >= n:
        return [[i] for i in range(n)]
    kmeans = faiss.Kmeans(embeddings.shape[1], n_clusters, niter=20, seed=42)
    kmeans.train(embeddings.astype("float32"))
    _, assignments = kmeans.index.search(embeddings.astype("float32"), 1)
    clusters: dict[int, list[int]] = {}
    for i, cluster_id in enumerate(assignments.flatten()):
        clusters.setdefault(int(cluster_id), []).append(i)
    return list(clusters.values())


def _summarize(generator: RAGGenerator, texts: list[str]) -> str:
    excerpts = "\n\n".join(f"- {t[:500]}" for t in texts)
    prompt = SUMMARY_PROMPT.format(excerpts=excerpts)
    config = types.GenerateContentConfig(max_output_tokens=SUMMARY_MAX_TOKENS)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
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
        except httpx.ConnectError:
            if attempt == MAX_TRANSIENT_RETRIES:
                raise
            time.sleep(TRANSIENT_RETRY_DELAY_SECONDS)
    return ""  # unreachable - loop always returns or raises


def build_raptor_tree(
    documents: list[dict],
    embedder: EmbeddingModel,
    generator: RAGGenerator,
    cluster_size: int = DEFAULT_CLUSTER_SIZE,
    max_levels: int = DEFAULT_MAX_LEVELS,
) -> list[RaptorNode]:
    """documents: [{chunk_id, text, metadata}] as returned by
    VectorStore.get_all_documents(). Returns every node across every level
    (leaves included) - this flat list is what gets persisted and searched."""
    all_nodes: list[RaptorNode] = []
    current_level_nodes = [
        RaptorNode(
            node_id=d["chunk_id"], level=0, text=d["text"],
            embedding=embedder.embed_query(d["text"]).tolist(),
            child_ids=[], source=d["metadata"].get("source", "unknown"),
        )
        for d in documents
    ]
    all_nodes.extend(current_level_nodes)

    for level in range(1, max_levels + 1):
        if len(current_level_nodes) < MIN_NODES_TO_CLUSTER:
            break
        embeddings = np.array([n.embedding for n in current_level_nodes])
        clusters = _cluster_indices(embeddings, cluster_size)
        if len(clusters) == len(current_level_nodes):
            break  # clustering collapsed to singletons - no more compression to gain

        next_level_nodes: list[RaptorNode] = []
        for member_indices in clusters:
            members = [current_level_nodes[i] for i in member_indices]
            summary_text = _summarize(generator, [m.text for m in members])
            if not summary_text:
                continue
            node = RaptorNode(
                node_id=str(uuid.uuid4()), level=level, text=summary_text,
                embedding=embedder.embed_query(summary_text).tolist(),
                child_ids=[m.node_id for m in members], source="summary",
            )
            next_level_nodes.append(node)
        if not next_level_nodes:
            break
        all_nodes.extend(next_level_nodes)
        current_level_nodes = next_level_nodes

    return all_nodes


def save_tree(nodes: list[RaptorNode], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(n) for n in nodes]))


def load_tree(path: str | Path) -> list[RaptorNode]:
    return [RaptorNode(**d) for d in json.loads(Path(path).read_text())]


class RaptorRetriever:
    """Collapsed-tree search: treats every node (leaf chunk or cluster
    summary, at any level) as an equally-searchable candidate, ranked by
    cosine similarity to the query. See module docstring for why."""

    def __init__(self, nodes: list[RaptorNode]):
        self.nodes = nodes
        self._embeddings = np.array([n.embedding for n in nodes]) if nodes else np.empty((0, 0))

    def search(
        self, query: str, embedder: EmbeddingModel, k: int = 10
    ) -> list[tuple[RaptorNode, float]]:
        if not self.nodes:
            return []
        query_embedding = embedder.embed_query(query)
        scores = self._embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.nodes[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    import shutil

    from core.chunking import chunk_pages
    from core.ingestion import DocumentLoader
    from core.vectorstore import ChromaStore

    smoke_dir = Path(__file__).parent.parent / "vectorstore_data" / "_smoke_raptor"
    shutil.rmtree(smoke_dir, ignore_errors=True)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    pages = DocumentLoader().load_directory(corpus_dir)[:8]
    chunks = chunk_pages(pages, strategy="recursive")
    store = ChromaStore(persist_dir=smoke_dir, collection_name="smoke_raptor")
    store.add_documents(chunks)

    embedder = EmbeddingModel()
    generator = RAGGenerator()
    documents = store.get_all_documents()
    print(f"Building RAPTOR tree over {len(documents)} leaf chunks...")
    nodes = build_raptor_tree(documents, embedder, generator, cluster_size=8, max_levels=2)
    by_level: dict[int, int] = {}
    for n in nodes:
        by_level[n.level] = by_level.get(n.level, 0) + 1
    print("Nodes per level:", by_level)

    retriever = RaptorRetriever(nodes)
    results = retriever.search("visual line of sight requirements", embedder, k=3)
    for node, score in results:
        print(f"  level={node.level} score={score:.3f} source={node.source} text={node.text[:100]!r}")

    shutil.rmtree(smoke_dir, ignore_errors=True)
