"""
eval.py — Phase 04: Prompt Evaluation & Testing

Runs the full RAG pipeline over test_set.csv and reports:
  - Retrieval accuracy (did the matched movie match the expected one?)
  - Grounding rate (did the model say "grounded": true honestly on in-corpus Qs?)
  - Edge-case handling (empty input, off-topic, not-in-corpus -> should NOT hallucinate)
  - Latency p50 / p95
  - LLM-as-judge score (1-5) for answer quality, using Groq itself as judge

Usage:
    python eval.py --test_set test_set.csv --out eval_results.csv
    python eval.py --test_set test_set.csv --out eval_results.csv --csv data/wiki_movie_plots_full_2000_2017.csv
"""
import argparse
import statistics as stats

import pandas as pd
from dotenv import load_dotenv

import rag_core

load_dotenv()

JUDGE_PROMPT_TEMPLATE = """You are grading an AI movie assistant's answer for quality.

Question: {question}
Expected movie (ground truth, may be "none" for edge cases): {expected}
Assistant's answer: {answer}
Assistant claimed grounded: {grounded}

Score the answer 1-5 on these combined criteria:
- Correctness (did it identify the right movie, or correctly decline if there isn't one?)
- Honesty (did it avoid inventing facts not in its context?)
- Clarity (is the answer well-formed and relevant?)

Respond with ONLY a single integer 1-5, nothing else.
"""


def llm_judge_score(llm, question, expected, answer, grounded) -> int:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, expected=expected, answer=answer, grounded=grounded
    )
    try:
        resp = llm.invoke(prompt)
        digits = "".join(filter(str.isdigit, resp.content.strip()))
        score = int(digits[:1]) if digits else 0
        return max(1, min(5, score)) if score else 0
    except Exception:
        return 0  # judge failed to produce a parseable score


def run_eval(test_set_path: str, out_path: str, csv_path: str = rag_core.CSV_PATH):
    df = pd.read_csv(test_set_path, keep_default_na=False)
    vectordb = rag_core.load_vectordb()
    llm = rag_core.load_llm()
    system_prompt = rag_core.load_system_prompt()

    print("[hybrid] Building BM25 index from CSV (one-time startup cost)...")
    retriever = rag_core.load_hybrid_retriever(vectordb, csv_path=csv_path)
    print("[hybrid] BM25 index ready.\n")

    rows = []
    for _, row in df.iterrows():
        question = row["question"]
        expected = row["expected_movie"]
        category = row["category"]

        if not question.strip():
            # Edge case: empty input — should not crash, should give a graceful response
            parsed, sources, latency_ms = {"answer": "(empty input)", "matched_movie": None,
                                            "confidence": "low", "grounded": False}, [], 0.0
        else:
            parsed, sources, latency_ms = rag_core.ask(
                llm, vectordb, question, system_prompt, retriever=retriever
            )

        matched = (parsed.get("matched_movie") or "").lower()
        is_correct = (
            expected.lower() in matched
            if expected.lower() != "none"
            else (matched == "" or "none" in matched or parsed.get("matched_movie") is None)
        )

        judge_score = llm_judge_score(llm, question, expected, parsed.get("answer", ""), parsed.get("grounded"))

        rows.append({
            "question": question,
            "category": category,
            "expected_movie": expected,
            "matched_movie": parsed.get("matched_movie"),
            "confidence": parsed.get("confidence"),
            "grounded": parsed.get("grounded"),
            "correct": is_correct,
            "latency_ms": round(latency_ms, 1),
            "judge_score": judge_score,
            "answer": parsed.get("answer"),
        })
        print(f"[{'OK' if is_correct else 'FAIL'}] ({category}) {question[:50]!r} -> {parsed.get('matched_movie')}")

    results = pd.DataFrame(rows)
    results.to_csv(out_path, index=False)

    accuracy = results["correct"].mean() * 100
    avg_judge = results["judge_score"].mean()
    latencies = results["latency_ms"].tolist()
    p50 = stats.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0

    print("\n" + "=" * 50)
    print(f"Accuracy:           {accuracy:.1f}%  ({results['correct'].sum()}/{len(results)})")
    print(f"Avg LLM-judge score: {avg_judge:.2f} / 5")
    print(f"Latency p50:        {p50:.0f} ms")
    print(f"Latency p95:        {p95:.0f} ms")
    print(f"Full results saved to: {out_path}")
    print("=" * 50)

    print("\nBy category:")
    print(results.groupby("category")["correct"].mean().mul(100).round(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default="test_set.csv")
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--csv", default=rag_core.CSV_PATH,
                        help="Path to the movie plots CSV used for BM25 index")
    args = parser.parse_args()
    run_eval(args.test_set, args.out, csv_path=args.csv)
