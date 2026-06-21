"""
app.py — Movie Plot Detective
Phases covered: 03 (prompting), 05 (RAG), 08 (backend logic), 09 (frontend/UX), 10 (guardrails, logging)

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub -> create a Hugging Face Space (Streamlit SDK) -> point it at the repo.
    Add GROQ_API_KEY as a Space secret (Settings -> Variables and secrets).
"""
import os
import csv
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import rag_core

load_dotenv()
LOG_PATH = "logs/call_log.csv"

def log_call(question: str, answer: dict, latency_ms: float, retrieved_sources: list):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "matched_movie", "confidence", "grounded", "latency_ms", "retrieved_sources"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            question,
            answer.get("matched_movie"),
            answer.get("confidence"),
            answer.get("grounded"),
            round(latency_ms, 1),
            "; ".join(retrieved_sources),
        ])


# ---------- Cached resources (loaded once per session) ----------
@st.cache_resource
def load_vectordb():
    return rag_core.load_vectordb()


@st.cache_resource
def load_llm():
    try:
        return rag_core.load_llm()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource
def load_retriever(_vectordb):
    """Build the BM25 index once and cache it for the session."""
    return rag_core.load_hybrid_retriever(_vectordb)


def ask(llm, vectordb, question: str, retriever):
    return rag_core.ask(llm, vectordb, question, retriever=retriever)


# ---------- UI ----------
st.set_page_config(page_title="Movie Plot Detective", page_icon="🎬")
st.title("🎬 Movie Plot Detective")
st.caption(
    "Ask about a movie plot, or describe a scene and I'll try to identify the film. "
    "Answers are grounded in a retrieval database — not general AI knowledge. "
    "**You are interacting with an AI assistant.**"
)

with st.sidebar:
    st.markdown("### About")
    st.write("RAG demo over a movie plot dataset. Retrieval: Chroma + local embeddings. Generation: Groq (Llama 3.3).")
    st.write("Dataset: see `db/` — rebuild with `python ingest.py`.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

vectordb = load_vectordb()
llm = load_llm()
retriever = load_retriever(vectordb)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])} · {msg.get('latency_ms', 0):.0f}ms")

example_cols = st.columns(3)
examples = [
    "A thief who plants ideas in dreams",
    "A family secretly living under a rich household",
    "A dinosaur theme park gone wrong",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.pending_input = ex

user_input = st.chat_input("Describe a plot or ask a movie question...") or st.session_state.pop("pending_input", None)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            result, sources, latency_ms = ask(llm, vectordb, user_input, retriever)
        st.write(result["answer"])
        badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.get("confidence") or "", "⚪")
        st.caption(f"{badge} Confidence: {result.get('confidence')} · Sources: {', '.join(sources) if sources else 'none'} · {latency_ms:.0f}ms")
        log_call(user_input, result, latency_ms, sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
        "latency_ms": latency_ms,
    })
