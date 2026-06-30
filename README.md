# Medicare Coverage Intelligence Platform

A retrieval system over CMS coverage policy and the clinical literature that resolves Medicare coverage the way it actually works — **national first, then jurisdictional** — and flags where coverage diverges from the evidence.

Two capabilities for provider revenue-cycle teams, both driven by one **NCD→LCD coverage cascade**:

1. **Policy Q&A** — "Is this covered, and under what criteria?" An NCD governs nationally; if there's no NCD (or it defers to local contractors), coverage falls back to the beneficiary's **MAC jurisdiction LCD**. For denial *prevention* at prior-auth.
2. **Evidence Gap Analysis** — "Where is coverage out of step with the published evidence?" The governing policy (NCD *or* LCD) is compared against PubMed, with a PMID-cited gap report. For denial *appeals*.

Routing is automatic; jurisdictional questions (mention a state) flow through the same cascade, which asks for the state when it needs one.

Prototype: https://medicare-coverage-agent.streamlit.app/

---

## The coverage cascade (Phase 3)

LCDs are **jurisdictional** (one MAC contractor per region) and too numerous/variable to treat like NCDs, so coverage resolves in a deterministic cascade — the shared resolution step both Policy Q&A and Gap Analysis call:

```
question (+ optional state)
  │
  ▼  retrieve + rerank NCDs
 Is there an NCD, and does it GOVERN?
  ├─ governs            → answer/gap from the national NCD                ✅ done
  └─ silent / defers-to-MAC
        │
        ▼  state → MAC  (deterministic table; ask the user if unknown)
     Resolve the beneficiary's MAC jurisdiction
        ├─ out-of-scope MAC → report (not guessed)
        └─ in-scope MAC
              ▼  RAG over that MAC's LCD bodies (jurisdiction-filtered)
           Is a relevant LCD found?
              ├─ yes → answer/gap from the live-indexed LCD
              └─ no  → "no determination → contractor discretion"
```

- **NCD-first, deterministic.** Whether an NCD governs is a relevance gate + a defer-marker check (`ncd_disposition`: governs / defers / silent). `state → MAC` is a static table. Only the final answer/gap reasoning is generative.
- **Jurisdiction-aware.** Scoped to **5 MACs** — Noridian, CGS, WPS, Palmetto, National Government Services (the best-represented). States served by out-of-scope MACs are reported as unsupported, never guessed.
- **Multi-turn.** If the cascade needs the beneficiary's state and the question didn't include one, it asks; the next message continues the query.

### LCDs are body-based RAG, not titles

An earlier design matched questions to LCD *titles* and fetched them live — it failed because **questions don't carry LCD titles** ("panniculectomy" vs an LCD titled "Plastic Surgery"). So LCDs are now **indexed like NCDs**: the indications/limitations **body** is chunked (title-prepend + synonym expansion), tagged with **boolean `mac_<key>` flags** (one doc per LCD, even when shared across MACs — no duplication). Retrieval is body-based semantic search **filtered to the beneficiary's MAC**, so a query reaches the right LCD by *content*. Retired LCDs are excluded (a retired determination can't drive coverage).

---

## Architecture

```
┌──────────────────────────┐        ┌──────────────────────────┐
│   CMS Coverage API       │        │   PubMed / NCBI          │
│   NCDs + LCDs (5 MACs,   │        │   E-utilities            │
│   active, with body)     │        │   (per NCD & LCD topic)  │
└────────────┬─────────────┘        └────────────┬─────────────┘
             ▼                                   ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│  cms_coverage (Chroma)   │        │  pubmed_evidence (Chroma)│
│  1,483 NCD + 10,668 LCD  │        │  5,735 abstracts         │
│  chunks; bge embeddings; │        │  (3,219 NCD + 2,516 LCD  │
│  title+synonym; mac flags│        │  topical, tier-graded)   │
└────────────┬─────────────┘        └────────────┬─────────────┘
             │                                   │
             ▼  resolve_governing_policy (NCD→LCD cascade)
   ┌──────────────────────────────────────────────────┐
   │  NCD governs?  → NCD                               │
   │  silent/defers → state→MAC → MAC-filtered LCD RAG  │
   │  neither       → none ("idk")                      │
   └───────────────┬───────────────────┬───────────────┘
                   ▼                    ▼
         ┌──────────────────┐  ┌──────────────────────────┐
         │  Policy Q&A      │  │  Gap Analysis            │
         │  answer from the │  │  governing policy (NCD   │
         │  governing policy│  │  or LCD) vs PubMed →     │
         │  (cited)         │  │  structured gap report   │
         └─────────┬────────┘  └────────────┬─────────────┘
                   ▼                         ▼
   ┌──────────────────────────────────────────────────────┐
   │  Streamlit UI (auto-routes Q&A vs gap; multi-turn     │
   │  state ask) · JSONL logs · eval harness               │
   └──────────────────────────────────────────────────────┘
```

### Model roles

| Model | Provider | Role |
|---|---|---|
| `gemini-2.5-flash` | Google AI | Policy answer + gap-report generation · NCD disambiguation · independent gap labeling · RAGAS judge |
| `llama-3.1-8b-instant` | Groq (free tier) | Synthetic question generation (Policy Q&A, gap, and LCD eval sets) |
| `BAAI/bge-small-en-v1.5` | HuggingFace (local) | Document/query embeddings (both collections) |
| `BAAI/bge-reranker-base` | HuggingFace (local) | Cross-encoder reranking — NCD selection, LCD relevance, governance gates |

---

## Evaluation

### Policy Q&A (RAGAS)

79-question NCD subset, config k=10 / threshold=0.65 / rerank top-5 / hybrid. Judge: `gemini-2.5-flash`.

| Metric | Score | Target |
|---|:---:|:---:|
| Faithfulness | 0.854 | > 0.90 |
| Answer Relevancy | 0.833 | > 0.85 |
| Context Precision | **0.917** ✅ | > 0.80 |
| Empty Retrieval | **0.0%** ✅ | < 15% |
| Citation Accuracy | **0.911** | > 95% |
| Policy Recall | **0.987** ✅ | > 90% |

### NCD gap analysis (independent judge)

Reference labels from an independent `gemini-2.5-flash` labeler (reads raw NCD + abstracts, never the pipeline's report), Claude cross-vendor adjudicated; 276-record golden set. `alignment_accuracy = label_match ∧ ncd_recall`, decomposing failures into retrieval vs reasoning. Full-277 whole-NCD baseline: `alignment_accuracy` 0.542, `kappa` 0.433, `faithfulness` 0.705, `citation_precision` 0.994. The current lean stack (disambiguation + tightened Partial + Insufficient-gate fix) measured **acc 0.520→0.640 on a paired random-25**; a full-277 confirmation run is the next eval.

### LCD coverage cascade (Phase 3)

`generate_golden_lcd.py` writes a lay coverage question per sampled LCD (12/MAC) and pairs it with a state in that jurisdiction; `judge_lcd.py` scores both **resolution** (which LCD the cascade picks — no API cost) and **generation faithfulness** (`--faithfulness`: RAGAS grounding of the answer in the LCD body, reference-free).

| Metric | Value |
|---|:---:|
| `lcd_precision` — right LCD when it picks one (n=60) | **0.94** |
| `lcd_recall` — resolved to the correct LCD (n=60) | **0.783** |
| disposition (n=60) | lcd 50 · ncd 7 · none 3 |
| `lcd_faithfulness` — answer grounded in the LCD body (n=15) | **0.655** |

7 of the "misses" are services with a national NCD (cascade correctly returns the NCD); excluding those, LCD-applicable recall is **0.887**. A gate sweep (`scripts/sweep_lcd_gates.py`, no API cost) validated the two soft gates: `SILENT_GATE`=0.60 (stable [0.58, 0.65]) and `LCD_GATE`=0.55 (precision-optimal). **Generation faithfulness 0.655 trails the NCD policy answer (0.854)** — LCD bodies are long and boilerplate-heavy, so answers extrapolate coverage criteria more; a real quality gap to tighten. **Known limit:** ~half the LCDs have broad titles ("Plastic Surgery"), so per-title evidence fetch yields nothing focused → those gaps return Insufficient.

```bash
python -m src.evaluation.judge           # RAGAS — Policy Q&A
python -m src.evaluation.judge_gap       # NCD gap analysis
python -m src.evaluation.generate_golden_lcd  # build the LCD eval set (Groq)
python -m src.evaluation.judge_lcd       # LCD cascade eval (no API cost)
```

---

## Project Structure

```
src/
├── ingestion/
│   ├── fetch.py              # CMS API: NCDs + LCDs (fetch_lcds_for_macs, MAC tagging)
│   └── fetch_pubmed.py       # PubMed via NCBI; per-NCD and per-LCD topical evidence
├── rag/
│   ├── embedder.py           # bge-small embeddings
│   ├── indexer.py            # cms_coverage build/load (NCD + LCD bodies, mac flags)
│   ├── pubmed_indexer.py     # pubmed_evidence build/load
│   └── pipeline.py           # retrieval, rerank, NCD disambiguation, jurisdiction
│                             #   routing, NCD→LCD cascade, answer(), gap_analysis()
├── evaluation/
│   ├── generate_golden.py / generate_golden_gap.py / generate_golden_lcd.py
│   └── judge.py / judge_gap.py / judge_lcd.py
└── ui/
    └── app.py                # Streamlit — auto-routes Q&A vs gap, multi-turn state

scripts/                      # eval/analysis: gap random-N eval, LCD gate sweep
data/   chroma/  (Git LFS)  → cms_coverage (1,483 NCD + 10,668 LCD) + pubmed_evidence (5,735)
```

---

## Setup

**Prerequisites:** Python 3.11+, Git LFS, a Google AI API key (paid tier recommended), a Groq key (eval generation only).

```bash
git lfs install && git clone <repo>     # LFS pulls the prebuilt index
python -m venv agent
agent\Scripts\activate                  # Windows  (source agent/bin/activate on *nix)
pip install -r requirements.txt
```

Create `.env` (or set Streamlit Cloud secrets):

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here     # eval question generation only
```

The vector index ships in the repo via **Git LFS** — no rebuild needed to run or deploy. To rebuild from scratch:

```bash
python -m src.ingestion.fetch                       # NCDs + LCDs
python -m src.ingestion.fetch_pubmed                # per-NCD evidence
python -m src.ingestion.fetch_pubmed --lcd          # per-LCD evidence
python -m src.rag.indexer                           # cms_coverage (NCD + LCD)
python -m src.rag.pubmed_indexer                    # pubmed_evidence
```

Launch:

```bash
streamlit run src/ui/app.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Policy answers · gap synthesis · NCD disambiguation · gap labeling · RAGAS judge |
| `GROQ_API_KEY` | Eval generation only | Synthetic question generation (Q&A / gap / LCD eval sets) |
