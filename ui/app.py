"""Streamlit chat UI for the RAG Technical Doc Assistant."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.citation_check import verify_citations
from core.compression import compress_chunks
from core.confidence import compute_confidence, should_refuse
from core.generation import RAGGenerator
from core.retrieval import DEFAULT_DENSE_K, DEFAULT_FINAL_K, RetrieverWithReranker
from core.vectorstore import ChromaStore

DEMO_QUESTIONS = [
    "What is the maximum altitude for drone operations in India?",
    "What are the registration requirements for UAS under DGCA?",
    "How does the FAA define visual line of sight?",
]

st.set_page_config(page_title="RAG Technical Doc Assistant", page_icon=":satellite:", layout="wide")


@st.cache_resource
def get_store() -> ChromaStore:
    return ChromaStore()


@st.cache_resource
def get_generator() -> RAGGenerator:
    return RAGGenerator()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def render_sidebar(store: ChromaStore) -> dict:
    st.sidebar.header("Corpus")
    stats = store.get_stats()
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Documents", stats["total_documents"])
    col2.metric("Chunks", stats["total_chunks"])
    with st.sidebar.expander("Source documents"):
        for source in stats["sources"]:
            st.write(f"- {source}")
    st.sidebar.caption("Active chunking strategy: recursive (rebuild index with "
                        "--strategy to compare fixed/semantic)")

    st.sidebar.header("Retrieval settings")
    final_k = st.sidebar.slider("Chunks used for generation (k)", 1, 10, DEFAULT_FINAL_K)
    dense_k = st.sidebar.slider("Dense candidates before rerank", 5, 50, DEFAULT_DENSE_K)
    use_reranker = st.sidebar.checkbox("Cross-encoder reranking", value=True)
    use_mmr = st.sidebar.checkbox("MMR diversity (overrides reranker)", value=False)

    st.sidebar.header("Advanced retrieval")
    use_hybrid = st.sidebar.checkbox("Hybrid (BM25 + dense, RRF)", value=False)
    use_query_expansion = st.sidebar.checkbox("Query expansion (extra LLM call)", value=False)
    use_hyde = st.sidebar.checkbox("HyDE (extra LLM call)", value=False)
    use_correction = st.sidebar.checkbox(
        "CRAG/agentic correction loop (overrides reranker toggle, extra LLM call on retry)",
        value=False,
    )
    expand_to_parent = st.sidebar.checkbox("Expand to parent section", value=False)
    use_compression = st.sidebar.checkbox("Extractive context compression", value=False)
    do_verify_citations = st.sidebar.checkbox("Verify citations", value=False)

    jurisdiction = st.sidebar.selectbox(
        "Jurisdiction filter", ["any", "india", "us", "n/a"], index=0
    )

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

    return {
        "final_k": final_k,
        "dense_k": dense_k,
        "use_reranker": use_reranker,
        "use_mmr": use_mmr,
        "use_hybrid": use_hybrid,
        "use_query_expansion": use_query_expansion,
        "use_hyde": use_hyde,
        "use_correction": use_correction,
        "expand_to_parent": expand_to_parent,
        "use_compression": use_compression,
        "do_verify_citations": do_verify_citations,
        "where": None if jurisdiction == "any" else {"jurisdiction": jurisdiction},
    }


def render_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        for s in sources:
            section_suffix = f', {s["section"]}' if s.get("section") else ""
            st.markdown(
                f"**{s['filename']}** — page {s['page']}{section_suffix} "
                f"(relevance: {s['relevance_score']:.2f})"
            )
            if s.get("chunk_text"):
                st.caption(s["chunk_text"][:500] + ("..." if len(s["chunk_text"]) > 500 else ""))


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])
                meta = message.get("metadata", {})
                st.caption(
                    f"Chunks used: {meta.get('chunks_used', '-')} | "
                    f"Tokens: {meta.get('tokens_used', '-')} | "
                    f"Latency: {meta.get('latency_ms', '-')} ms"
                )


def handle_query(query: str, store: ChromaStore, generator: RAGGenerator, settings: dict) -> None:
    retriever = RetrieverWithReranker(store)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        start = time.time()
        if settings["use_mmr"]:
            chunks = retriever.retrieve_mmr(query, dense_k=settings["dense_k"], final_k=settings["final_k"])
        elif settings["use_correction"]:
            chunks, _iterations = retriever.retrieve_with_correction(
                query, dense_k=settings["dense_k"], final_k=settings["final_k"],
                use_hybrid=settings["use_hybrid"], where=settings["where"],
                expand_to_parent=settings["expand_to_parent"],
            )
        else:
            chunks = retriever.retrieve(
                query, dense_k=settings["dense_k"], final_k=settings["final_k"],
                use_reranker=settings["use_reranker"], use_hybrid=settings["use_hybrid"],
                use_query_expansion=settings["use_query_expansion"], use_hyde=settings["use_hyde"],
                where=settings["where"], expand_to_parent=settings["expand_to_parent"],
            )
        if settings["use_compression"]:
            chunks = compress_chunks(query, chunks, store.embedder)

        placeholder = st.empty()
        full_text = ""
        for delta in generator.stream_generate(
            query, chunks, conversation_history=st.session_state.conversation_history
        ):
            full_text += delta
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        latency_ms = int((time.time() - start) * 1000)
        response = generator.last_response

        confidence = compute_confidence(chunks[0] if chunks else None)
        refused = should_refuse(confidence)

        citations = None
        if settings["do_verify_citations"] and chunks:
            citations = [
                v.to_dict()
                for v in verify_citations(full_text, chunks, retriever._get_cross_encoder())
            ]

        sources = [
            {**src.to_dict(), "chunk_text": chunk.text}
            for src, chunk in zip(response.sources, chunks)
        ]
        render_sources(sources)
        st.caption(
            f"Chunks used: {response.chunks_used} | "
            f"Tokens: {response.tokens_used} | "
            f"Latency: {latency_ms} ms | "
            f"Confidence: {confidence:.2f}{' (low - consider refusing)' if refused else ''}"
        )
        if citations:
            with st.expander("Citation verification"):
                for c in citations:
                    if c["supported"] is None:
                        continue
                    icon = "✅" if c["supported"] else "⚠️"
                    st.caption(f"{icon} {c['sentence']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": sources,
            "metadata": {
                "chunks_used": response.chunks_used,
                "tokens_used": response.tokens_used,
                "latency_ms": latency_ms,
                "confidence": confidence,
            },
        }
    )
    st.session_state.conversation_history.append({"role": "user", "content": query})
    st.session_state.conversation_history.append({"role": "assistant", "content": full_text})


def main() -> None:
    init_session_state()
    store = get_store()
    generator = get_generator()
    settings = render_sidebar(store)

    st.title("RAG Technical Doc Assistant")
    st.caption("Ask questions about drone/UAS regulations grounded in DGCA, FAA, and RPAS technical documents.")

    selected_demo = None
    if not st.session_state.messages:
        st.write("Try asking:")
        cols = st.columns(len(DEMO_QUESTIONS))
        for col, demo_question in zip(cols, DEMO_QUESTIONS):
            if col.button(demo_question):
                selected_demo = demo_question

    render_history()

    user_input = st.chat_input("Ask a question about drone regulations...")
    query = selected_demo or user_input
    if query:
        handle_query(query, store, generator, settings)


if __name__ == "__main__":
    main()
