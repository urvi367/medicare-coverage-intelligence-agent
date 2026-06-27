# Medicare Coverage Intelligence Platform

A retrieval system over CMS coverage policy and the clinical literature that answers two questions for provider revenue-cycle teams:

1. **"What does Medicare cover, and under what criteria?"** — grounded, cited answers from CMS National and Local Coverage Determinations (NCDs/LCDs), for **denial *prevention*** at prior-auth time.
2. **"Where is CMS coverage out of step with the published evidence?"** — a structured, PMID-cited gap report comparing each NCD against PubMed, for **denial *appeals*** ("not medically necessary" / "experimental").

The UI auto-routes each question to the right path by regex signal scoring — no manual mode toggle.

Prototype: https://medicare-coverage-agent.streamlit.app/

---

## Architecture

```
┌──────────────────────────┐        ┌──────────────────────────┐
│   CMS Coverage API       │        │   PubMed / NCBI          │
│   NCDs + LCDs            │        │   E-utilities            │
└────────────┬─────────────┘        └────────────┬─────────────┘
             │ fetch (8-worker pool,             │ two-pass per-NCD search
             │ iterative HTML unescape)          │ (primary-evidence first)
             ▼                                   ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│  cms_coverage  (Chroma)  │        │  pubmed_evidence (Chroma)│
│  1,983 NCD/LCD chunks    │        │  3,219 abstracts /       │
│  bge-small embeddings    │        │  296 NCD topics, T1–T5   │
│  title + synonym expand  │        │  tier-graded, ~89% RCT/  │
│                          │        │  meta/cohort             │
└────────────┬─────────────┘        └────────────┬─────────────┘
             │                                   │
   ┌─────────┴──────────┐                        │
   ▼                    ▼                        │
┌─────────────────┐  ┌───────────────────────────┴──────────────────┐
│  POLICY Q&A     │  │              GAP ANALYSIS                      │
│  (prevention)   │  │              (appeals)                        │
├─────────────────┤  ├───────────────────────────────────────────────┤
│ hybrid retrieve │  │ NCD-only hybrid retrieve + cross-encoder rerank│
│ (BM25+dense,RRF)│  │            │                                   │
│   │             │  │            ▼  score-weighted vote per NCD      │
│   ▼             │  │  ┌──────────────────────────────────────────┐ │
│ bge-reranker    │  │  │ NCD disambiguation: when top candidates  │ │
│ top-5           │  │  │ score close, one LLM call reads the      │ │
│   │             │  │  │ question vs candidate titles → picks the │ │
│   ▼             │  │  │ governing NCD(s), cap 2                   │ │
│ cited context   │  │  └──────────────────────┬───────────────────┘ │
│ + LCD juris note│  │            ▼  whole-NCD full text (not chunks) │
│   │             │  │     topical join → PubMed for that NCD only   │
│   ▼             │  │            ▼                                   │
│ gemini-2.5-flash│  │     gemini-2.5-flash → structured gap report  │
│ → cited answer  │  │     (Coverage Position · Evidence · Grade ·   │
│                 │  │      Alignment · Gap Summary)                 │
└────────┬────────┘  └───────────────────────┬───────────────────────┘
         │                                   │
         ▼                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Streamlit UI (auto-routes Q&A vs gap) · JSONL interaction logs   │
│  Evaluation: judge.py (RAGAS, Policy Q&A) · judge_gap.py (gap)    │
└──────────────────────────────────────────────────────────────────┘
```

### Model roles

| Model | Provider | Role |
|---|---|---|
| `gemini-2.5-flash` | Google AI | Policy answer generation · **gap report synthesis** · **NCD disambiguation** · independent gap reference labeling · RAGAS judge |
| `llama-3.1-8b-instant` | Groq | Synthetic question generation (Policy Q&A + gap golden sets) |
| `BAAI/bge-small-en-v1.5` | HuggingFace (local) | Document/query embeddings (both collections) + RAGAS Answer Relevancy |
| `BAAI/bge-reranker-base` | HuggingFace (local) | Cross-encoder reranking; on the gap side, used only to *identify* the governing NCD |

---

## Gap Analysis — how it works

Beyond surface retrieval, the gap pipeline reasons about **where CMS coverage and the evidence diverge**, and maps each verdict to one analyst action.

- **Governing-NCD selection (disambiguation).** NCD-only hybrid retrieval + cross-encoder rerank produce a score-weighted vote per NCD. When the top candidates score close — the case where the rerank argmax is unreliable — **one bounded LLM call reads the question against the candidate NCD titles and picks which policy actually governs** (cap 2). This corrects the primary pick (top-1 accuracy 0.830 → 0.917 on the golden set) and selects a single clean policy, lifting recall *and* cutting cross-policy contamination — something neither a score threshold nor a numbering-hierarchy rule could do.
- **Whole-NCD context.** The full text of the governing NCD is supplied — not the top-5 question-similar chunks — so the eligibility-criteria section (which separates a *Partial Coverage Gap* from *Aligned*) can never be dropped by chunk ranking.
- **Topical join for evidence.** PubMed abstracts are pulled *only for the governing NCD(s)* (`source_ncd_number == policy_number`), so evidence and coverage position describe the same intervention. Dense top-12, newest-first, no reranker (bge-reranker isn't trained on clinical text). An empty join yields *Insufficient Evidence*, never unrelated abstracts.
- **Structured synthesis.** `gemini-2.5-flash` emits a fixed format — CMS Coverage Position · Clinical Evidence (`PMID` bullets, tier-graded T1–T5) · Evidence Grade · Alignment · Gap Summary. The prompt forbids citing un-retrieved PMIDs.
- **Calibrated label boundaries.** *Partial* requires a **named** excluded population treated with the **same** covered intervention (a different drug/device/program is not this policy's gap). The *Insufficient* gate discards an abstract only as a **name-collision** — a genuinely different intervention sharing a name — never for studying an adjacent population of the same intervention.
- **Action-oriented alignment.** Aligned (no action) · Partial Coverage Gap (broaden) · Coverage Gap (expand/appeal) · Overcoverage (utilization review) · Insufficient Evidence (manual review).

---

## Evaluation

### Policy Q&A (RAGAS)

79-question NCD subset (LCDs filtered). Config: k=10, threshold=0.65, reranker top_n=5, hybrid search. Judge: `gemini-2.5-flash`.

| Metric | Score | Target |
|---|:---:|:---:|
| Faithfulness | 0.854 | > 0.90 |
| Answer Relevancy | 0.833 | > 0.85 |
| Context Precision | **0.917** ✅ | > 0.80 |
| Empty Retrieval | **0.0%** ✅ | < 15% |
| Citation Accuracy | **0.911** | > 95% |
| Policy Recall | **0.987** ✅ | > 90% |

### Gap analysis (independent judge)

Reference labels come from an **independent `gemini-2.5-flash` labeler** that reads the raw NCD + abstracts — never the pipeline's own report — and are additionally cross-vendor adjudicated by Claude. 276-record golden set.

`alignment_accuracy = label_match ∧ ncd_recall`, so failures decompose into *retrieval* vs *reasoning*.

| Metric | Measures |
|---|---|
| `alignment_accuracy` | End-to-end: right alignment **and** right NCD retrieved |
| `alignment_label_match` | Raw label agreement (retrieval-blind) |
| `alignment_kappa` | Quadratic-weighted Cohen's κ on the evidence-vs-coverage direction |
| `alignment_action_match` | Provider action bucket: appeal / covered / manual |
| `ncd_recall` | Governing NCD surfaced and selected |
| `pmid_recall` / `pmid_recall_retrieved` | Reference-PMID citation recall (overall / over retrieved) |
| `citation_precision` | Cited PMIDs that were actually retrieved (fabrication guard) |
| `faithfulness` | RAGAS faithfulness vs policy + PubMed contexts |

**Where it stands.** The last full 277-record benchmark of the whole-NCD pipeline scored `alignment_accuracy` 0.542, `kappa` 0.433, `faithfulness` 0.705, `ncd_recall` 0.888 (`citation_precision` 0.994 = effectively zero fabrication). The current build then added three changes — the **disambiguation step**, a tightened **Partial** boundary, and an **Insufficient-gate** precision fix:

- NCD selection (LLM-free backtest, n=277): top-1 **0.830 → 0.917**, `ncd_recall` **0.888 → 0.921**, multi-policy contamination **26% → 0.7%**.
- Paired representative sample (seeded-random 25, same records old vs new): `alignment_accuracy` **0.520 → 0.640**, `ncd_recall` **0.800 → 0.880**.

A full 277-record re-eval of the current stack is the next step; the random-25 predicts it clears the prior 0.570 baseline.

```bash
python -m src.ingestion.fetch_pubmed          # fetch PubMed abstracts
python -m src.rag.pubmed_indexer              # build pubmed_evidence collection
python -m src.evaluation.generate_golden_gap  # independent reference labels
python -m src.evaluation.judge_gap            # score the gap pipeline
```

---

## Project Structure

```
src/
├── ingestion/
│   ├── fetch.py              # CMS Coverage API client — NCDs + LCDs
│   └── fetch_pubmed.py       # PubMed via NCBI E-utilities (two-pass primary-evidence)
├── rag/
│   ├── embedder.py           # HuggingFace embedding wrapper
│   ├── indexer.py            # ChromaDB build/load — cms_coverage collection
│   ├── pubmed_indexer.py     # ChromaDB build/load — pubmed_evidence collection
│   └── pipeline.py           # Hybrid retrieval, rerank, NCD disambiguation,
│                             #   generation, gap_analysis
├── evaluation/
│   ├── generate_golden.py        # Synthetic Policy Q&A dataset
│   ├── generate_golden_gap.py    # Independent gap reference labels
│   ├── judge.py                  # RAGAS evaluation — Policy Q&A
│   └── judge_gap.py              # Gap analysis evaluation
└── ui/
    └── app.py                # Streamlit UI — auto-routes Q&A vs gap

scripts/                      # one-off analysis: NCD-selection sweeps + backtests,
                              #   paired random-N gap eval

data/   chroma/  → cms_coverage (1,983 chunks) + pubmed_evidence (3,219 abstracts)
        (golden_gap.json, gap_questions.json, pubmed_raw.json are gitignored)
logs/   eval_*.jsonl/json, answer caches, interactions.jsonl  (gitignored)
```

---

## Setup

**Prerequisites:** Python 3.11+, a Google AI API key (paid tier recommended).

```bash
# 1. virtual environment
python -m venv agent
agent\Scripts\activate        # Windows
source agent/bin/activate     # macOS / Linux

# 2. dependencies
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here   # only for question generation
```

**Build the policy index (Policy Q&A):**

```bash
python -m src.ingestion.fetch       # fetch NCDs + LCDs (~2–3 min)
python -m src.rag.indexer           # chunk, embed, persist cms_coverage
```

**Build the evidence index (Gap Analysis):**

```bash
python -m src.ingestion.fetch_pubmed   # fetch abstracts per NCD topic
python -m src.rag.pubmed_indexer       # persist pubmed_evidence
```

**Launch the UI:**

```bash
streamlit run src/ui/app.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Policy answers · gap synthesis · NCD disambiguation · gap labeling · RAGAS judge |
| `GROQ_API_KEY` | Question generation only | Synthetic Policy Q&A + gap question sets |
```
