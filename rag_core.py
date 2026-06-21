"""
rag_core.py — Shared RAG pipeline logic (retrieval + generation + guardrails).
Kept separate from app.py so eval.py can run headless without Streamlit.

Phase 06 update: Hybrid retrieval (BM25 on full plots + vector search on chunks)
using Reciprocal Rank Fusion to improve paraphrase and detail query accuracy.
"""
import os
import re
import math
import json
import time
from pydantic import SecretStr

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "db"
TOP_K = 8
PROMPT_PATH = "prompts/system_prompt_v2_final.txt"
CSV_PATH = "data/wiki_movie_plots_full_2000_2017.csv"

# BM25 config
BM25_TOP_CANDIDATES = 5  # how many movies BM25 nominates
BM25_CHUNKS_PER_MOVIE = 2  # how many Chroma chunks to fetch per BM25 candidate
BM25_SCORE_THRESHOLD = 8.0  # minimum BM25 score to trust a candidate

BLOCKLIST_TOPICS = ["politics", "religion", "suicide", "self-harm", "weapon", "bomb"]
FALLBACK_RESPONSE = {
    "answer": "I can only help with movie plots and trivia from this dataset — I'm not able to help with that.",
    "matched_movie": None,
    "confidence": "low",
    "sources": [],
    "grounded": False,
}


# ---------------------------------------------------------------------------
# Lightweight BM25 (no external deps beyond what's already installed)
# ---------------------------------------------------------------------------
class _BM25Index:
    """Fast in-memory BM25 index built from the movie plots CSV."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        n = len(corpus)
        doc_freqs: dict[str, int] = {}
        doc_len: list[int] = []
        doc_tfs: list[dict[str, int]] = []
        inverted: dict[str, list[int]] = {}

        total_len = 0
        for idx, doc in enumerate(corpus):
            tokens = self._tok(doc)
            doc_len.append(len(tokens))
            total_len += len(tokens)
            tfs: dict[str, int] = {}
            for t in tokens:
                tfs[t] = tfs.get(t, 0) + 1
            doc_tfs.append(tfs)
            for t in tfs:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
                inverted.setdefault(t, []).append(idx)

        self.avgdl = total_len / n if n else 1.0
        self.doc_len = doc_len
        self.doc_tfs = doc_tfs
        self.inverted = inverted
        self.idf: dict[str, float] = {
            t: math.log((n - f + 0.5) / (f + 0.5) + 1.0)
            for t, f in doc_freqs.items()
        }
        self._n = n

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def get_scores(self, query: str) -> list[float]:
        scores = [0.0] * self._n
        for t in self._tok(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for idx in self.inverted.get(t, []):
                tf = self.doc_tfs[idx][t]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * num / den
        return scores


# ---------------------------------------------------------------------------
# Hybrid Retriever — built once, reused across queries
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Combines:
    - BM25 on full plot summaries (from CSV) to identify the right movie
    - Chroma vector search on chunks for detailed context
    Then merges results, BM25 candidates first.
    """

    def __init__(self, vectordb, csv_path: str = CSV_PATH):
        self.vectordb = vectordb

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["plot"])
        df = df.drop_duplicates(subset=["title"])

        self._titles: list[str] = df["title"].tolist()

        # Prepend title + year to plot so exact title queries score higher
        corpus = []
        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            year = str(row.get("year", ""))
            plot = str(row.get("plot", ""))
            corpus.append(f"{title} {year} {plot}")

        self._bm25 = _BM25Index(corpus)

    def _bm25_candidates(self, query: str) -> list[str]:
        """Return up to BM25_TOP_CANDIDATES movie titles with high BM25 scores."""
        scores = self._bm25.get_scores(query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :BM25_TOP_CANDIDATES
        ]
        return [
            self._titles[i]
            for i in top_idx
            if scores[i] >= BM25_SCORE_THRESHOLD
        ]

    def retrieve(self, query: str, k: int = TOP_K) -> list:
        """
        Return up to k LangChain Document objects.
        BM25-nominated chunks come first; vector chunks fill the remainder.
        """
        candidate_titles = self._bm25_candidates(query)

        # Step 1 — fetch BM25 candidate chunks from Chroma (filtered by title)

        per_title_results = {}
        for title in candidate_titles:
            per_title_results[title] = self.vectordb.similarity_search(
                query, k=BM25_CHUNKS_PER_MOVIE, filter={"title": title}
            )
        bm25_docs = []
        for round_idx in range(BM25_CHUNKS_PER_MOVIE):
            for title in candidate_titles:
                chunks = per_title_results[title]
                if round_idx < len(chunks):
                    bm25_docs.append(chunks[round_idx])

      # Step 2 — standard vector search
        vector_docs = self.vectordb.similarity_search(query, k=k)

        # Step 3 — merge with a guaranteed split, so BM25 candidates can never
        # fully crowd out semantic vector results (or vice versa).
        bm25_budget = k // 2
        vector_budget = k - bm25_budget

        seen: set[str] = set()
        merged = []

        for doc in bm25_docs[:bm25_budget]:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                merged.append(doc)

        for doc in vector_docs:
            if len(merged) >= k:
                break
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                merged.append(doc)
                vector_budget -= 1

        # If vector search didn't fill its budget (e.g. too many duplicates),
        # backfill remaining slots with leftover BM25 candidates.
        for doc in bm25_docs[bm25_budget:]:
            if len(merged) >= k:
                break
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                merged.append(doc)

        return merged[:k]
    
    


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_system_prompt(path: str = PROMPT_PATH) -> str:
    with open(path) as f:
        return f.read()


def load_vectordb(persist_dir: str = PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def load_hybrid_retriever(vectordb, csv_path: str = CSV_PATH) -> HybridRetriever:
    """Build the BM25 index and return a ready-to-use HybridRetriever.
    Call once at startup; reuse the returned object for all queries.
    """
    return HybridRetriever(vectordb, csv_path=csv_path)


def load_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your free key "
            "from https://console.groq.com/keys"
        )
    return ChatGroq(model=model, api_key=SecretStr(api_key), temperature=temperature)


def is_blocked(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(topic in lowered for topic in BLOCKLIST_TOPICS)


def retrieve_context(vectordb, query: str, k: int = TOP_K, retriever: HybridRetriever | None = None):
    """Retrieve context chunks.
    
    If a HybridRetriever is provided, uses hybrid BM25 + vector search.
    Falls back to plain vector search if no retriever is given (backward compat).
    """
    if retriever is not None:
        results = retriever.retrieve(query, k=k)
    else:
        results = vectordb.similarity_search(query, k=k)

    sources = [doc.metadata.get("source", "unknown") for doc in results]
    context_text = "\n\n".join(
        f"[{doc.metadata.get('source')}]: {doc.page_content}" for doc in results
    )
    return context_text, sources


def ask(llm, vectordb, question: str, system_prompt: str | None = None,
        retriever: HybridRetriever | None = None):
    """Runs one full RAG turn: guardrail check -> retrieve -> generate -> parse.
    Returns (parsed_response_dict, sources_list, latency_ms).

    Pass a HybridRetriever (from load_hybrid_retriever()) for improved accuracy.
    """
    if is_blocked(question):
        return dict(FALLBACK_RESPONSE), [], 0.0

    if system_prompt is None:
        system_prompt = load_system_prompt()

    start = time.time()
    context_text, sources = retrieve_context(vectordb, question, retriever=retriever)
    prompt = system_prompt.format(context=context_text, question=question)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "answer": "Sorry, I had trouble forming a structured answer. Please try rephrasing the question.",
            "matched_movie": None,
            "confidence": "low",
            "sources": [],
            "grounded": False,
        }

    latency_ms = (time.time() - start) * 1000
    return parsed, sources, latency_ms
