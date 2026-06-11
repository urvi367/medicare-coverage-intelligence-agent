# Medicare Coverage Intelligence Agent

A RAG system that answers natural-language questions about Medicare coverage policy by retrieving directly from CMS National Coverage Determinations (NCDs) and Local Coverage Determinations (LCDs). Answers are grounded, cited, and evaluated end-to-end.

Prototype: https://medicare-coverage-agent.streamlit.app/
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
│  • Iterative HTML unescape (fixes CMS double-escaping)          │
│  • 8-worker ThreadPoolExecutor for parallel detail fetching     │
│  • Saved to data/ncd_raw.json, data/lcd_raw.json               │
└───────────────────────────┬─────────────────────────────────────┘
                            │  load_documents()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Indexing Layer                              │
│  • RecursiveCharacterTextSplitter (800 chars / 100 overlap)     │
│  • Blank chunk filter — no empty vectors in index               │
│  • Title prepend + synonym expansion on every chunk             │
│  • BAAI/bge-small-en-v1.5  — local CPU embeddings, no API cost  │
│  • ChromaDB persisted at data/chroma/ (wiped on rebuild)        │
└──────────────┬────────────────────────────┬─────────────────────┘
               │  dense (k=10, t=0.65)      │  BM25 sparse (k=10)
               ▼                            ▼
┌──────────────────────────────────────────────────────────────── ┐
│              Hybrid Retrieval — Reciprocal Rank Fusion          │
│  • Dense vector search catches semantic similarity              │
│  • BM25 catches exact term/numeric matches (e.g. "55 mmHg")     │
│  • RRF (k=60) fuses both ranked lists → up to 20 candidates     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  merged candidates
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Cross-Encoder Reranker                        │
│  • BAAI/bge-reranker-base — scores (query, chunk) pairs jointly │
│  • Selects top 5 — real filtering, not just reordering          │
└───────────────────────────┬─────────────────────────────────────┘
                            │  top-5 reranked chunks
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  • Constructs cited context block with NCD/LCD headers          │
│  • LCD jurisdiction note injected only when top-ranked doc      │
│    is an LCD (not on stray LCD chunks)                          │
│  • gemini-2.5-flash — answer generation (temperature=0)         │
│  • Indefinite retry loop reading API retryDelay on rate limits  │
│  • Returns answer + source Documents                            │
└──────────┬────────────────────────────┬─────────────────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────┐   ┌─────────────────────────────────────┐
│    Streamlit UI     │   │         Evaluation Pipeline         │
│  • Chat interface   │   │                                     │
│  • Source expander  │   │  generate_golden.py                 │
│    with full chunk  │   │  • llama-3.1-8b-instant (Groq)      │
│    text per source  │   │  • 198 Q&A pairs from policy docs   │
│  • Feedback buttons │   │  • Incremental save + resume        │
│  • JSONL logging    │   │                                     │
│    of interactions  │   │  judge.py (RAGAS)                   │
└─────────────────────┘   │  • Faithfulness                     │
                          │  • Answer Relevancy                 │
                          │  • Context Precision                │
                          │  • gemini-2.5-flash as judge        │
                          │  • BAAI/bge-small-en-v1.5 embeddings│
                          │  • Cache path derived from config   │
                          │  • policy_recall + citation_accuracy│
                          │  • NCD-only eval (LCD filtered)     │
                          └─────────────────────────────────────┘
```

### Model roles

| Model | Provider | Role |
|---|---|---|
| `gemini-2.5-flash` | Google AI | Answer generation in the RAG pipeline + RAGAS judge |
| `llama-3.1-8b-instant` | Groq | Synthetic Q&A generation (golden dataset only) |
| `BAAI/bge-small-en-v1.5` | HuggingFace (local) | Document and query embeddings + RAGAS Answer Relevancy embeddings |
| `BAAI/bge-reranker-base` | HuggingFace (local) | Cross-encoder reranking of retrieved chunks |

---

## Evaluation Results

Evaluated on 79 NCD questions (LCD entries excluded — Phase 1). Config: k=10, threshold=0.65, reranker top_n=5. Judge: gemini-2.5-flash.

| Metric | Score | Target |
|---|:---:|:---:|
| Faithfulness | 0.821 | > 0.90 |
| Answer Relevancy | **0.857** ✅ | > 0.85 |
| Context Precision | **0.892** ✅ | > 0.80 |
| Empty Retrieval | **1.3%** ✅ | < 15% |
| Citation Accuracy | 0.886 | > 95% |
| Policy Recall | **0.962** ✅ | > 90% |

---

## Project Structure

```
src/
├── ingestion/
│   └── fetch.py              # CMS API client — NCDs and LCDs
├── rag/
│   ├── embedder.py           # HuggingFace embedding wrapper
│   ├── indexer.py            # ChromaDB build + load (cms_coverage collection)
│   └── pipeline.py           # Hybrid retrieval + reranking + Gemini generation
├── evaluation/
│   ├── generate_golden.py    # Synthetic dataset generation (198 pairs)
│   └── judge.py              # RAGAS evaluation pipeline
└── ui/
    └── app.py                # Streamlit chat interface

data/                         # committed to git
├── ncd_raw.json              # Raw NCD data from CMS API
├── lcd_raw.json              # Raw LCD data from CMS API
└── chroma/                   # ChromaDB — cms_coverage collection (1983 chunks)

data/                         # gitignored
├── pubmed_raw.json           # PubMed abstracts (Phase 2 — branch: phase-2)
└── golden_dataset.json       # 198-pair evaluation set (79 NCD, 119 LCD)

logs/                         # gitignored
├── eval_results.jsonl        # Aggregate scores per run
├── eval_samples_latest.json  # Per-sample scores from latest run
├── rag_answers_cache_*.json  # Persistent answer cache (named by config + search_mode)
└── interactions.jsonl        # UI interaction log
```

---

## Setup

**Prerequisites:** Python 3.11+, a Google AI API key (paid tier recommended).

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
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here   # only needed for generate_golden.py
```

### 4. Ingest CMS data

Fetches all NCDs and LCDs from the CMS Coverage API (~2–3 min):

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

Generates 198 synthetic Q&A pairs (79 NCD, 119 LCD) using `llama-3.1-8b-instant` via Groq. Supports resuming — re-running picks up where it left off:

```bash
python -m src.evaluation.generate_golden
```

### Run RAGAS evaluation

Scores the pipeline on the 79-question NCD subset. Answers are cached in `logs/rag_answers_cache_*.json` — already-answered questions are never re-fetched:

```bash
python -m src.evaluation.judge
```

Per-sample scores are written to `logs/eval_samples_latest.json` after each run.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI API key — answer generation + RAGAS judge |
| `GROQ_API_KEY` | Only for `generate_golden.py` | Groq API key for synthetic dataset generation |
