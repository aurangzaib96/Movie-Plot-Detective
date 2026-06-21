"""
ingest.py — Phase 05 (RAG): Data prep + embedding + vector store.

Loads a movie dataset (CSV with columns: title, year, genre, plot),
chunks long plots, embeds with a local sentence-transformers model
(free, no API calls, no rate limits), and stores everything in a
persistent Chroma vector database.

Usage:
    python ingest.py --csv data/sample_movies.csv --persist_dir db/
    python ingest.py --csv data/wiki_movie_plots.csv --persist_dir db/   # swap in the real Kaggle dataset
"""
import argparse
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs on CPU


def load_movies(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"title", "plot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    # Basic cleaning: drop empty plots, dedupe, cap dataset size for free-tier speed
    df = df.dropna(subset=["plot"])
    df = df.drop_duplicates(subset=["title"])
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Turn each movie row into a Document with rich metadata for citation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = []
    for _, row in df.iterrows():
        title = str(row.get("title", "Unknown"))
        year = row.get("year", "n/a")
        genre = row.get("genre", "n/a")
        plot = str(row["plot"])

        chunks = splitter.split_text(plot)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "year": year,
                        "genre": genre,
                        "chunk_index": i,
                        "source": f"{title} ({year})",
                    },
                )
            )
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample_movies.csv")
    parser.add_argument("--persist_dir", default="db")
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Max rows to embed — keeps free-tier embedding fast. Increase once you've confirmed the pipeline works.",
    )
    args = parser.parse_args()

    print(f"[1/4] Loading {args.csv} ...")
    df = load_movies(args.csv)
    if len(df) > args.limit:
        print(f"  -> Dataset has {len(df)} rows, capping to {args.limit} for speed (use --limit to change).")
        df = df.head(args.limit)
    print(f"  -> {len(df)} movies loaded.")

    print("[2/4] Chunking plots ...")
    docs = build_documents(df)
    print(f"  -> {len(docs)} chunks created.")

    print(f"[3/4] Embedding with local model '{EMBEDDING_MODEL}' (no API calls, free) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[4/4] Storing in Chroma at '{args.persist_dir}' ...")
    os.makedirs(args.persist_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=args.persist_dir,
    )
    vectordb.persist()
    print(f"Done. Vector store ready at '{args.persist_dir}/' with {len(docs)} chunks.")


if __name__ == "__main__":
    main()
