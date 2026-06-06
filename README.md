# Medicare Coverage Intelligence Agent

A production-ready RAG system that answers natural-language questions about Medicare coverage policy by retrieving directly from CMS National Coverage Determinations (NCDs) and Local Coverage Determinations (LCDs). Answers are grounded, cited, and evaluated end-to-end.

---

## Why this exists

Navigating Medicare coverage policy is painful. Clinicians and patients searching for whether a procedure is covered must cross-reference dense PDF documents across hundreds of NCDs and LCDs. This agent indexes the full CMS coverage corpus and lets users ask plain-English questions, receiving cited answers in seconds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CMS Coverage API                         │
│          api.coverage.cms.gov/v1  (NCDs + LCDs)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │  fetch_and_save() — parallel HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Ingestion Layer                             │
│  • HTML stripping + text normalization                          │
│  • 8-worker ThreadPoolExecutor for parallel detail fetching     │
│  • LCD license Bearer token auth                                │
│  • Saved to data/ncd_raw.json, data/lcd_raw.json               │
└───────────────────────────┬─────────────────────────────────────┘
                            │  load_documents()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Indexing Layer                              │
│  • RecursiveCharacterTextSplitter (800 chars / 100 overlap)     │
│  • BAAI/bge-small-en-v1.5  — local CPU embeddings, no API cost  │
│  • ChromaDB persisted at data/chroma/                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │  similarity search (k=5)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  • Retrieves top-5 policy chunks                                │
│  • Constructs cited context block                               │
│  • llama-3.3-70b-versatile (Groq) — answer generation          │
│  • Returns answer + source Documents                            │
└──────────┬────────────────────────────────┬─────────────────────┘
           │                                │
           ▼                                ▼
┌─────────────────────┐       ┌─────────────────────────────────────┐
│    Streamlit UI      │       │         Evaluation Pipeline          │
│  • Chat interface   │       │                                     │
│  • Source expander  │       │  generate_golden.py                 │
│  • JSONL logging    │       │  • llama-3.1-8b-instant generates   │
│    of interactions  │       │    200 Q&A pairs from policy docs   │
└─────────────────────┘       │  • Incremental save + resume        │
                              │  • Rate-limit retry w/ retry-after  │
                              │                                     │
                              │  judge.py (RAGAS)                   │
                              │  • Faithfulness                     │
                              │  • Answer Relevancy                 │
                              │  • llama-3.1-8b-instant as judge    │
                              └─────────────────────────────────────┘
```

### Model roles

| Model | Provider | Role |
|---|---|---|
| `llama-3.3-70b-versatile` | Groq | Answer generation in the RAG pipeline |
| `llama-3.1-8b-instant` | Groq | Synthetic Q&A generation + LLM-as-judge (RAGAS) |
| `BAAI/bge-small-en-v1.5` | HuggingFace (local) | Document and query embeddings |

The 70b model is used where answer quality matters most. The 8b model handles high-volume, repetitive tasks (200 synthetic generations + per-question RAGAS scoring) to stay within free-tier token limits.

---

## Project Structure

```
src/
├── ingestion/
│   └── fetch.py          # CMS API client — NCDs and LCDs
├── rag/
│   ├── embedder.py       # HuggingFace embedding wrapper
│   ├── indexer.py        # ChromaDB build + load
│   └── pipeline.py       # Retrieval + Groq generation
├── evaluation/
│   ├── generate_golden.py  # Synthetic dataset generation (200 pairs)
│   └── judge.py            # RAGAS faithfulness + answer relevancy
└── ui/
    └── app.py            # Streamlit chat interface

data/                     # gitignored — generated at runtime
├── ncd_raw.json
├── lcd_raw.json
├── chroma/               # ChromaDB vector store
└── golden_dataset.json   # 200-pair evaluation set

logs/                     # gitignored — JSONL interaction logs
tests/
```

---

## Setup

**Prerequisites:** Python 3.11+, a [Groq API key](https://console.groq.com) (free tier works).

### 1. Create and activate the virtual environment

```bash
python -m venv agent
agent\Scripts\activate      # Windows
source agent/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Ingest CMS data

Fetches all NCDs and LCDs from the CMS Coverage API (parallel, ~2–3 min):

```bash
python -m src.ingestion.fetch
```

### 5. Build the vector index

Chunks, embeds, and persists documents to ChromaDB:

```bash
python -m src.rag.indexer
```

### 6. Launch the UI

```bash
streamlit run src/ui/app.py
```

---

## Evaluation

### Generate the golden dataset

Generates 200 synthetic Q&A pairs from sampled policy documents using `llama-3.1-8b-instant`. Supports resuming across Groq free-tier daily token limits — re-running picks up exactly where it left off:

```bash
python -m src.evaluation.generate_golden
```

### Run RAGAS evaluation

Scores the RAG pipeline on faithfulness and answer relevancy against the golden dataset:

```bash
python -m src.evaluation.judge
```

Output example:
```
faithfulness: 0.847
answer_relevancy: 0.912
```

---

## Groq Free Tier Notes

The pipeline is designed to work within Groq's free tier limits:

| Model | RPM | TPM | TPD |
|---|---|---|---|
| `llama-3.3-70b-versatile` | 30 | 12K | 100K |
| `llama-3.1-8b-instant` | 30 | 6K | 500K |

- `generate_golden.py` enforces a **10s delay** between requests and uses `retry-after` headers on 429s
- `judge.py` enforces a **5s delay** between RAG pipeline calls during evaluation
- Golden dataset generation is split across days if needed via incremental save

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for both generation and evaluation |
