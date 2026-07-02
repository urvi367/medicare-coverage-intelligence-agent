# Medicare Coverage Intelligence Platform

**A production-style RAG system that resolves Medicare coverage the way CMS actually works — national policy first, then jurisdictional — and flags where coverage has fallen behind the clinical evidence.**

Built for provider revenue-cycle teams: prevent denials at prior-auth, and build evidence-backed appeals after them. Retrieval over **CMS NCDs/LCDs + PubMed**, a **deterministic NCD→LCD coverage cascade**, and an **independent LLM-judge evaluation harness** at every stage.

🔗 **Live prototype:** https://medicare-coverage-agent.streamlit.app/

---

## Highlights

- **Domain-faithful retrieval architecture** — coverage resolves through a **deterministic NCD→LCD cascade** (national determination first; fall back to the beneficiary's MAC-jurisdiction LCD), not a flat vector search. The generative step is the *last* step; routing is gates + a static state→MAC table.
- **Two products, one resolver** — Policy Q&A (*"is this covered, under what criteria?"*) and Evidence-Gap Analysis (*"where does coverage diverge from the literature?"*) both call the same `resolve_governing_policy`.
- **Hybrid retrieval done properly** — BM25 + dense vectors fused with Reciprocal Rank Fusion, then a cross-encoder reranker; whole-document policy context so eligibility criteria can't be lost to chunking.
- **Evaluation as a first-class citizen** — RAGAS for Q&A, an **independent cross-vendor LLM judge** for gap analysis (labels never derived from the pipeline's own output), and a no-API cascade-resolution eval. Metrics decompose failures into *retrieval vs reasoning*.
- **Engineering judgment on display** — several designs were built, measured, and **rejected** (live title-matching, an agentic tool-use loop, a heuristic evidence filter, a prompt-gate). The README documents *why*, with the eval numbers that killed them.
- **Grounded, cited, honest** — every answer cites its policy/PMID sources; zero fabricated citations in eval; known limitations are stated, not hidden.

---

## What it does

| Capability | Question | User | Driven by |
|---|---|---|---|
| **Policy Q&A** | "Is this covered, and under what criteria?" | Prior-auth / denial-*prevention* | NCD→LCD cascade → cited answer from the governing policy |
| **Evidence Gap Analysis** | "Where is coverage out of step with the evidence?" | Denial-*appeals* specialists | Governing policy (NCD *or* LCD) vs PubMed → PMID-cited gap report |

Routing is automatic (regex signal-scoring, no mode toggle). Jurisdictional questions flow through the same cascade, which asks for the beneficiary's state only when it needs one.

---

## The coverage cascade

LCDs are **jurisdictional** — one MAC contractor per region, the *same* service covered differently across MACs, ~969 final LCDs — so they can't be treated like national NCDs. Coverage resolves in a deterministic cascade, the shared resolution step both capabilities call:

```
question (+ optional state)
  │
  ▼  retrieve + rerank NCDs
 Is there an NCD, and does it GOVERN?          (relevance gate + ncd_disposition)
  ├─ governs            → answer/gap from the national NCD                ✅
  └─ silent / defers-to-MAC
        │
        ▼  state → MAC   (deterministic table; ask the user if unknown)
     Resolve the beneficiary's MAC jurisdiction
        ├─ out-of-scope MAC → report as unsupported (never guessed)
        └─ in-scope MAC
              ▼  RAG over that MAC's LCD bodies (jurisdiction-filtered)
           Is a relevant LCD found?
              ├─ yes → answer/gap from the indexed LCD
              └─ no  → "no determination → contractor discretion"
```

- **Deterministic where it must be.** Whether an NCD governs is a relevance gate + defer-marker check (`ncd_disposition`: governs / defers / silent). `state → MAC` is a static table. Only the final answer/gap reasoning is generative.
- **Jurisdiction-aware.** Scoped to **5 MACs** (Noridian, CGS, WPS, Palmetto, National Government Services). States served by out-of-scope MACs are reported unsupported, never guessed.
- **Multi-turn.** If a jurisdictional question omits the state, the cascade asks; the next message continues the query.

---

## Architecture

End-to-end: **ingestion → dual Chroma stores → the deterministic cascade → the two capabilities → UI/eval.** Local (CPU) components carry no API cost; only the final generation and the eval judge call an LLM.

```
                         INGESTION  (offline; refresh + re-index)
┌───────────────────────────────────────┐   ┌───────────────────────────────────────────────┐
│  CMS Coverage API  (Bearer token)     │   │  PubMed / NCBI  E-utilities  (3 req/s throttle │
│  • NCDs  (benefit_category, is_lab)   │   │   + retry/backoff, _eutils_get)                │
│  • LCDs  (body, retired-filter,       │   │  • per-NCD topic search  (_search_topic_pmids) │
│    contractor→MAC)                    │   │  • per-LCD topic search  (_search_lcd_pmids)   │
│  • LCD→Article→CPT codes  (hcpc-code) │   │  • DIAGNOSTIC ANCHOR on test policies:         │
│        → data/lcd_diagnostic.json     │──▶│    NCD via benefit_category, LCD via CPT map   │
└──────────────────┬────────────────────┘   │    (search the TEST, not the analyte/disease)  │
                   │  fetch.py                └───────────────────┬───────────────────────────┘
                   ▼  indexer.py                                  ▼  fetch_pubmed.py → pubmed_indexer.py
┌───────────────────────────────────────┐   ┌───────────────────────────────────────────────┐
│  cms_coverage  (Chroma)               │   │  pubmed_evidence  (Chroma)                     │
│  1,483 NCD + 10,668 LCD chunks        │   │  ~7,179 abstracts, tier-graded (T1–T5)         │
│  bge-small embeddings · title+synonym │   │  topical join keys: source_ncd_number /        │
│  boolean mac_<key> flags (1 doc/LCD)  │   │  source_lcd_number  (evidence ↔ same policy)   │
└──────────────────┬────────────────────┘   └───────────────────┬───────────────────────────┘
                   │                                             │
   QUERY ──────────┼─────────────────────────────────────────────┼──────────────────────────
                   ▼   resolve_governing_policy(question, state)  │  (shared by both capabilities)
   ┌──────────────────────────────────────────────────────────┐  │
   │  1. hybrid retrieve+rerank NCDs (BM25+dense RRF, x-enc)   │  │
   │  2. NCD governs?  relevance ≥ SILENT_GATE  ∧              │  │
   │       ncd_disposition = governs                          │  │
   │       ├─ yes → NCD  (whole-doc context)                  │  │
   │       └─ silent / defers-to-MAC                          │  │
   │            3. state → MAC   (deterministic table)        │  │
   │               ├─ out-of-scope → none (unsupported)       │  │
   │               └─ in-scope → MAC-filtered LCD body RAG    │  │
   │                    ├─ top ≥ LCD_GATE → LCD (whole-doc)   │  │
   │                    └─ else → none (contractor discretion)│  │
   └───────────────┬──────────────────────────┬──────────────┘  │
                   ▼                           ▼                 │
         ┌──────────────────┐   ┌──────────────────────────────┐│
         │  Policy Q&A       │   │  Gap Analysis                ││ topical PubMed join
         │  answer() — cited │   │  gap_analysis() — governing  │◀┘  (_pubmed_for_ncds /
         │  from governing   │   │  policy (NCD|LCD) vs PubMed  │    _pubmed_for_lcd)
         │  policy body      │   │  → PMID-cited gap report     │
         └─────────┬─────────┘   └──────────────┬───────────────┘
                   ▼                            ▼   gemini-2.5-flash (temp 0)
   ┌──────────────────────────────────────────────────────────────┐
   │  Streamlit UI  (auto-routes Q&A vs gap · multi-turn state ask) │
   │  JSONL interaction/feedback logs  ·  eval harness (RAGAS +     │
   │  independent gap judge + LCD cascade eval)                     │
   └──────────────────────────────────────────────────────────────┘
```

### Why LCDs are body-based RAG, not title-matching

An earlier design matched questions to LCD *titles* and fetched them live — it failed because **questions don't carry LCD titles** ("panniculectomy" vs an LCD titled "Plastic Surgery"). LCDs are now **indexed like NCDs**: the indications/limitations **body** is chunked (title-prepend + synonym expansion), tagged with **boolean `mac_<key>` flags** (one doc per LCD even when shared across MACs — no duplication), and retrieved by *content*, filtered to the beneficiary's MAC. Retired LCDs are excluded — a retired determination can't drive coverage.

### Stack

| Layer | Choice | Why |
|---|---|---|
| Generation / judge | `gemini-2.5-flash` (temp 0) | Policy + gap synthesis · NCD disambiguation · independent gap labeling · RAGAS judge |
| Question generation | `llama-3.1-8b-instant` (Groq, free) | Synthetic eval-set questions only |
| Embeddings | `BAAI/bge-small-en-v1.5` (local CPU) | Same model at index + query time; zero API cost |
| Reranker | `BAAI/bge-reranker-base` (local CPU) | Cross-encoder — NCD selection, LCD relevance, governance gates |
| Vector store | ChromaDB (2 collections) | `cms_coverage` + `pubmed_evidence`; shipped via Git LFS |
| Retrieval | BM25 + dense + RRF (k=60) | Sparse handles exact terms/numerics ("55 mmHg"); dense handles semantics |

---

## Key engineering decisions

The interesting part isn't the happy path — it's what was tried, measured, and cut. Each was validated (or killed) with the eval harness.

- **Live title-matching → body-based RAG.** A Gemini function-calling loop fetched LCDs live and matched them by title. It couldn't work — questions never carry LCD titles. Replaced with content retrieval over indexed LCD bodies; the live-lookup tool and orchestrator were deleted.
- **Agentic loop → deterministic cascade.** The NCD→LCD flow is deterministic (gates + a static table), so a function-calling agent was over-engineering. The cascade lives in `pipeline.py` beside the NCD logic; the only place a reasoning call is justified is a narrow "which NCD governs?" disambiguation.
- **Chunk-level → whole-document policy context.** Top-5 chunk retrieval dropped the eligibility-criteria section that distinguishes a *Partial Coverage Gap* from *Aligned* (e.g. CPAP's AHI criteria). Switched to identifying the governing NCD (score-weighted vote, capped at 2) and feeding its full text → faithfulness +0.11, kappa +0.09, Partial off the floor.
- **Diagnostic-test evidence anchoring.** Test policies have bare analyte titles ("Magnesium", "HbA1c"), so a plain PubMed search returns *analyte-as-therapy* studies — right topic, wrong subject — which over-credited them in gap analysis. Test policies are detected **statically** (NCD `benefit_category`; LCD via CPT codes fetched through `lcd/related-documents → article/hcpc-code`) and their search is anchored on the test's own diagnostic accuracy. Regression-swept before shipping; honest about where it *doesn't* help (bare analytes with therapy-dominated literature; proprietary MolDX assays whose brand names don't match PubMed).
- **A "faithfulness problem" that was a measurement artifact.** LCD answers scored 0.65 RAGAS faithfulness vs 0.85 for NCD — because the LCD prompt *mandates* a jurisdiction disclaimer that's true but absent from the LCD body, so RAGAS flagged it ungrounded on every answer. Disclaimer-excluded, grounding is **0.843 ≈ NCD 0.854** — the model wasn't inventing criteria; the metric was.
- **Rejected: a heuristic evidence filter** (drop research-protocol boilerplate from policy context) — it *reliably* regressed the CPAP Partial gap across four runs even while retaining the criteria chunk. Verdicts are context-composition-sensitive; trimming must be section-aware and eval-validated, not heuristic.
- **Infra hardening found along the way** — CMS double-escapes HTML entities (`&amp;gt;`), so `_strip_html` unescapes to a fixed point; a Chroma rebuild silently 4×-duplicated the index (append-on-rebuild); the PubMed date parser hit ElementTree's childless-element-is-falsy gotcha (99% blank years); the LCD PubMed fetch had no throttle and swallowed 429s (mass-empty under load) — now a global rate-limiter with retry/backoff.

---

## Evaluation

Every stage has an eval; metrics are chosen to *decompose failures*, not just score them.

### Policy Q&A — RAGAS (79-question NCD set · k=10 / threshold=0.65 / rerank top-5 / hybrid)

| Metric | Score | Target |
|---|:---:|:---:|
| Context Precision | **0.917** ✅ | > 0.80 |
| Policy Recall | **0.987** ✅ | > 0.90 |
| Empty Retrieval | **0.0%** ✅ | < 15% |
| Citation Accuracy | 0.911 | > 0.95 |
| Faithfulness | 0.854 | > 0.90 |
| Answer Relevancy | 0.833 | > 0.85 |

### NCD gap analysis — independent judge

Reference labels come from an **independent `gemini-2.5-flash` labeler** (reads the raw NCD + abstracts, never the pipeline's report), then cross-vendor adjudicated by Claude — breaking the circularity of grading a model against its own output. 276-record golden set. `alignment_accuracy = label_match ∧ ncd_recall`, so high-label-match + low-recall ⇒ retrieval is the bottleneck; the reverse ⇒ reasoning is. Full-277 whole-NCD baseline: `alignment_accuracy` 0.542 · `kappa` 0.433 · `faithfulness` 0.705 · `citation_precision` 0.994 (zero fabrication).

### LCD gap analysis — independent judge

`judge_gap_lcd.py` mirrors the NCD gap eval through the cascade (n=28; `expected_lcd` + `state`), adding routing metrics (`disposition`, `lcd_recall`, `ncd_intercept`). After fixing an evidence-source mismatch (the golden had been labeled from a *different* PubMed fetch than the index held — a subtle evidence↔label coupling bug): **alignment_accuracy 0.32→0.61 · kappa 0→0.5 · faithfulness 0.72→0.76 · lcd_recall 0.929 · citation_precision 1.0**.

### LCD coverage cascade — no-API resolution eval

`judge_lcd.py` scores which LCD the cascade resolves to (no LLM cost) plus reference-free generation faithfulness. **`lcd_precision` 0.94 · `lcd_recall` 0.783** (n=60; LCD-applicable recall **0.887** after excluding NCD-intercepts). A no-API gate sweep validated the two soft gates (`SILENT_GATE`=0.60, `LCD_GATE`=0.55).

> ⏳ **Pending eval — diagnostic anchor.** The diagnostic-test anchoring is live in the index and the diagnostic golden records (8 NCD, 7 LCD) were re-labeled from the anchored evidence (4 flips Insufficient→Aligned). The paid gap-eval **re-run to quantify the anchor's effect is deferred** — re-labeling and index are done; only the LLM regeneration remains.

```bash
python -m src.evaluation.judge            # RAGAS — Policy Q&A
python -m src.evaluation.judge_gap        # NCD gap analysis
python -m src.evaluation.judge_gap_lcd    # LCD gap analysis
python -m src.evaluation.judge_lcd        # LCD cascade resolution (no API cost)
```

---

## Project structure

```
src/
├── ingestion/
│   ├── fetch.py            # CMS API: NCDs + LCDs, MAC tagging, LCD→Article→CPT diagnostic map
│   └── fetch_pubmed.py     # PubMed via NCBI: per-NCD/LCD topical evidence, diagnostic anchor, throttle+retry
├── rag/
│   ├── embedder.py         # bge-small embeddings
│   ├── indexer.py          # cms_coverage build/load (NCD + LCD bodies, mac flags)
│   ├── pubmed_indexer.py   # pubmed_evidence build/load
│   └── pipeline.py         # retrieval, rerank, NCD disambiguation, jurisdiction routing,
│                           #   NCD→LCD cascade, answer(), gap_analysis()
├── evaluation/
│   ├── generate_golden*.py # golden sets: Q&A, NCD gap, LCD gap, LCD cascade
│   └── judge*.py           # judge / judge_gap / judge_gap_lcd / judge_lcd
└── ui/app.py               # Streamlit — auto-routes Q&A vs gap, multi-turn state

data/ chroma/  (Git LFS)  → cms_coverage (1,483 NCD + 10,668 LCD) + pubmed_evidence (~7,179)
scripts/                  → eval/analysis: gap random-N eval, LCD gate sweep, backtests
```

---

## Setup

**Prerequisites:** Python 3.11+, Git LFS, a Google AI API key (paid tier recommended), a Groq key (eval generation only).

```bash
git lfs install && git clone <repo>     # LFS pulls the prebuilt index — no rebuild to run
python -m venv agent
agent\Scripts\activate                  # Windows  (source agent/bin/activate on *nix)
pip install -r requirements.txt
```

Create `.env` (or set Streamlit Cloud secrets):

```
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here     # eval question generation only
```

The vector index ships via **Git LFS** — no rebuild needed to run or deploy. To rebuild from scratch:

```bash
python -m src.ingestion.fetch                  # NCDs + LCDs
python -m src.ingestion.fetch_pubmed           # per-NCD evidence
python -m src.ingestion.fetch_pubmed --lcd     # per-LCD evidence
python -m src.rag.indexer                      # cms_coverage
python -m src.rag.pubmed_indexer               # pubmed_evidence
streamlit run src/ui/app.py
```
