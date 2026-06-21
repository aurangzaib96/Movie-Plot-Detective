# 🎬 Movie Plot Detective

A narrow-domain RAG assistant that identifies movies from plot descriptions and
answers movie-trivia questions — grounded only in a retrieved dataset, with
citations, confidence scores, and guardrails against hallucination.

Built for: Generative AI Engineer — AI-Driven Software Development project (SE 5th Semester).

## 1. Problem Statement

Generic chatbots answer movie questions from parametric memory, which means
they can confidently hallucinate plots, years, or cast details. **Movie Plot
Detective** instead retrieves the actual plot summary before answering, cites
its source, and explicitly says "I don't know" when the dataset doesn't cover
the question — solving a narrow, verifiable problem rather than being a
general-purpose chatbot.

**Target users:** film trivia communities, students building movie-recommendation
side projects, anyone who wants verifiable (not hallucinated) movie answers.

**Success metric:** retrieval accuracy on a held-out test set + honesty rate
(does it decline rather than invent answers on out-of-corpus questions?).

**Scope boundary — this app does NOT:** give real-time showtimes, box office
data, or recommend movies outside the ingested dataset.

## 2. Architecture

```
 User
   |
   v
Streamlit UI (app.py) ---- chat input, source citations, confidence badge
   |
   v
rag_core.ask()
   |-- guardrail check (block off-topic / unsafe input)
   |-- retrieve_context()  -> Chroma vector DB (local, persisted in db/)
   |                          embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free)
   |-- prompt template (prompts/system_prompt_v2_final.txt)
   |-- ChatGroq.invoke()   -> Groq API (Llama 3.3 70B, free tier)
   |-- JSON parse + fallback on malformed output
   v
Structured response: {answer, matched_movie, confidence, sources, grounded}
   |
   v
logs/call_log.csv  (latency, confidence, grounded — for monitoring)
```

## 3. Model & Tool Choices (with justification)

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` | Free tier, no card required, very low latency — important since free-tier rate limits punish slow iteration |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Zero API calls for embedding = zero rate-limit risk during ingestion, fully free |
| Vector DB | Chroma (local, persisted to disk) | No server/account needed, fine for a dataset this size |
| Orchestration | LangChain | Standard, well-documented, easy to swap providers later |
| Frontend | Streamlit | Fastest to build a working chat UI with streaming/loading states in one day |
| Deployment | Hugging Face Spaces | Free hosting, native Streamlit support |

Considered alternatives: OpenAI GPT-4o (rejected — no free tier), Pinecone
(rejected — Chroma is sufficient for this dataset size and avoids signup
friction), fine-tuning (rejected — out of scope for a 1-day free-tier build;
see Limitations).

## 4. Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your free Groq key from https://console.groq.com/keys

# Build the vector store (run once, or whenever the dataset changes)
python ingest.py --csv data/sample_movies.csv --persist_dir db

# Run the app
streamlit run app.py

# Run evaluation
python eval.py
```

### Swapping in a real dataset (e.g. Kaggle "Wikipedia Movie Plots")
1. Download the CSV from Kaggle.
2. Make sure it has at minimum a `title` and `plot` column (rename columns if needed — `year` and `genre` are optional but improve citations).
3. `python ingest.py --csv data/your_dataset.csv --persist_dir db`
4. Re-run `streamlit run app.py` — it will pick up the new `db/`.

## 5. Evaluation

`test_set.csv` has 20 questions across 4 categories: direct plot description,
detail questions, paraphrased descriptions, and edge cases (empty input,
off-topic, not-in-corpus). `eval.py` reports accuracy, an LLM-as-judge score
(1-5), and p50/p95 latency — see `eval_results.csv` after running it.

## 6. Guardrails & Responsible AI

- Off-topic / unsafe-topic keyword guardrail with a fixed fallback response (no LLM call wasted on blocked input)
- Prompt explicitly instructs the model to decline rather than invent an answer when context is insufficient
- Every response carries a `grounded` flag and a `confidence` level shown in the UI
- AI disclosure shown directly in the app ("You are interacting with an AI assistant")
- All LLM calls logged (latency, confidence, grounded) to `logs/call_log.csv` for monitoring

## 7. Limitations

- Dataset is a small hand-written sample by default — swap in the full Kaggle
  dataset for production-scale coverage (see Setup).
- No fine-tuning — prompt engineering + RAG was sufficient for this scope and
  avoided the dataset/compute requirements fine-tuning would need.
- Free-tier Groq rate limits mean the app may need to wait/retry under heavy
  concurrent load — see the checklist's own warning about testing under load
  before a live demo.
- LLM-as-judge scoring uses the same model family as the assistant, which can
  correlate errors; an independent judge model would be more rigorous.

## 8. Demo script (3 prepared scenarios)
1. "A thief who plants ideas in people's dreams" → should return Inception with high confidence and a citation.
2. "What's the weather today?" → should politely decline (guardrail).
3. "Tell me about Avengers Endgame" → should say it's not in the dataset rather than inventing a plot.
