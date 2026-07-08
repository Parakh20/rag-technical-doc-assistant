"""Streamlit chat UI for the RAG Technical Doc Assistant.

Thin HTTP/SSE client of the FastAPI service (api/main.py) - no direct
imports of core/pipeline retrieval code. Run the API first:
    uvicorn api.main:app --reload
then this UI:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

DEMO_QUESTIONS = [
    "What is the maximum altitude for drone operations in India?",
    "What are the registration requirements for UAS under DGCA?",
    "How does the FAA define visual line of sight?",
]

DEFAULT_DENSE_K = 20
DEFAULT_FINAL_K = 5

st.set_page_config(page_title="RAG Technical Doc Assistant", page_icon=":satellite:", layout="wide")


@st.cache_data(ttl=30)
def get_corpus_stats() -> dict:
    resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()["corpus"]


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def render_sidebar() -> dict:
    st.sidebar.header("Corpus")
    try:
        stats = get_corpus_stats()
    except requests.RequestException as exc:
        st.sidebar.error(f"API unreachable at {API_BASE_URL}: {exc}")
        stats = {"total_documents": "-", "total_chunks": "-", "sources": []}
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Documents", stats["total_documents"])
    col2.metric("Chunks", stats["total_chunks"])
    with st.sidebar.expander("Source documents"):
        for source in stats["sources"]:
            st.write(f"- {source}")

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
                    f"Latency: {meta.get('latency_ms', '-')} ms | "
                    f"Confidence: {meta.get('confidence', '-')}"
                )


def stream_query(payload: dict):
    """Yields ('delta', text) chunks, then a final ('done', response_dict)."""
    with requests.post(f"{API_BASE_URL}/query", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if "delta" in event:
                yield "delta", event["delta"]
            elif "error" in event:
                yield "error", event["error"]
            elif event.get("done"):
                yield "done", event["final"]


def handle_query(query: str, settings: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        start = time.time()
        payload = {
            "query": query,
            "dense_k": settings["dense_k"],
            "final_k": settings["final_k"],
            "use_reranker": settings["use_reranker"],
            "use_mmr": settings["use_mmr"],
            "use_hybrid": settings["use_hybrid"],
            "use_query_expansion": settings["use_query_expansion"],
            "use_hyde": settings["use_hyde"],
            "use_correction": settings["use_correction"],
            "where": settings["where"],
            "expand_to_parent": settings["expand_to_parent"],
            "use_compression": settings["use_compression"],
            "do_verify_citations": settings["do_verify_citations"],
            "conversation_history": st.session_state.conversation_history,
        }

        placeholder = st.empty()
        full_text = ""
        final_response = None
        try:
            for kind, value in stream_query(payload):
                if kind == "delta":
                    full_text += value
                    placeholder.markdown(full_text + "▌")
                elif kind == "error":
                    st.error(f"Generation failed: {value}")
                elif kind == "done":
                    final_response = value
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            return
        placeholder.markdown(full_text)

        latency_ms = int((time.time() - start) * 1000)
        response = final_response or {}
        sources = response.get("sources", [])
        confidence = response.get("confidence")
        refused = response.get("refused", False)
        citations = response.get("citations")

        render_sources(sources)
        confidence_str = f"{confidence:.2f}" if confidence is not None else "-"
        st.caption(
            f"Chunks used: {response.get('chunks_used', '-')} | "
            f"Tokens: {response.get('tokens_used', '-')} | "
            f"Latency: {latency_ms} ms | "
            f"Confidence: {confidence_str}{' (low - consider refusing)' if refused else ''}"
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
                "chunks_used": response.get("chunks_used"),
                "tokens_used": response.get("tokens_used"),
                "latency_ms": latency_ms,
                "confidence": confidence,
            },
        }
    )
    st.session_state.conversation_history.append({"role": "user", "content": query})
    st.session_state.conversation_history.append({"role": "assistant", "content": full_text})


def main() -> None:
    init_session_state()
    settings = render_sidebar()

    st.title("RAG Technical Doc Assistant")
    st.caption("Ask questions about drone/UAS regulations grounded in DGCA, FAA, and RPAS technical documents.")
    st.caption(f"API: {API_BASE_URL}")

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
        handle_query(query, settings)


if __name__ == "__main__":
    main()
