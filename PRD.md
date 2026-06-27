# Medicare Coverage Intelligence Platform — PRD

**Product:** Medicare Coverage Intelligence Platform
**Domain:** Provider Revenue Cycle — Prior Authorization & Denial Management
**Primary user:** Provider-side RCM/UM team — denial *prevention* pre-service (Phase 1) and denial *appeals* post-denial (Phase 2)
**Data sources:** CMS NCDs/LCDs (Phase 1) · PubMed (Phase 2)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-03-01 | Initial PRD — Phase 1 architecture with Groq/llama |
| v1.1 | 2026-05-01 | Updated to reflect live implementation |
| v1.2 | 2026-06-07 | Switch to gemini-2.5-flash/flash-lite; persistent answer cache; LCD jurisdiction prompt instruction; LCD entries filtered from eval; rate-limit retry using API retryDelay |
| v1.3 | 2026-06-08 | Fix AnswerRelevancy NaN (BAAI embeddings + bypass_n=True); upgrade to paid API tier; add eval results history |
| v1.4 | 2026-06-08 | Add cosine similarity threshold 0.7; per-sample scores to `logs/eval_samples_latest.json` |
| v1.5 | 2026-06-09 | Add cross-encoder reranker (bge-reranker-base, top 3 of k=5); conditionalize LCD boilerplate (inject only when LCD retrieved); fix NCD-only eval bug (legacy LCD cache entries were contaminating scores); remove redundant `document_type` field from golden dataset |
| v1.6 | 2026-06-09 | Fix `_strip_html` bug (HTML entities unescaped after tag-strip, leaving `<p>` noise in all chunks). Add title prepend + synonym expansion (50+ clinical↔CMS term pairs) to every chunk at index time. Context Precision target met (0.832). |
| v1.7 | 2026-06-10 | Widen retrieval to k=10 / threshold=0.65 (reranker now selects 5-of-10, not just reorders). Dynamic answer cache path derived from PIPELINE_CONFIG — each config gets its own file, no stale reuse. Add JUDGE_MODEL + PROMPT_TAG constants to judge.py; upgrade judge to gemini-2.5-flash. Add policy_recall metric. |
| v1.8 | 2026-06-10 | Fix root cause of persistent HTML entities: CMS data is double-escaped (`&amp;gt;` → `&gt;` after one pass). `_strip_html` now unescapes to fixed point. Fix tag regex to spare clinical comparisons (`< 80 mm Hg`). Fix `build_index` silently appending on rebuild (Chroma assigns fresh IDs — index was 8473 chunks / 4× duplication). Add `shutil.rmtree` wipe before rebuild. Fix LCD jurisdiction addendum firing on stray LCD chunks ranked 2nd–5th — now only fires when top-ranked doc is an LCD. First clean baseline established. |
| v1.9 | 2026-06-10 | Phase 2 scaffold: `src/ingestion/fetch_pubmed.py` (NCBI E-utilities, per-NCD topic search, dedup by PMID) + `src/rag/pubmed_indexer.py` (separate `pubmed_evidence` Chroma collection, safe rebuild). Moved to `phase-2` branch. |
| v2.0 | 2026-06-10 | Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (RRF k=60). BM25 fixes numeric threshold retrieval failures (e.g. "55 mmHg"). `search_mode` field added to PIPELINE_CONFIG; cache path now includes mode suffix. Chroma DB object cached in `_get_db()` — no longer reopened per query. Blank chunk filter added to `build_index()`. UI: feedback buttons (positive/partial/negative), full chunk text in sources expander. Deployed to Streamlit Community Cloud (`main` branch). |
| v2.2 | 2026-06-13 | **Gap eval run + hardening.** 284-record reference fully Claude-adjudicated (67 overrides/24%). Fixed name-collision bug (topical join pulled same-name-different-thing abstracts) via disambiguation in `_GAP_SYSTEM`/`_LABEL_PROMPT`; `LABEL_MODEL` flash-lite→flash. Captured PubMed PublicationType for grounded Evidence Grade. First 40-sample eval then two fixes: topic-forward question regen (`ncd_recall` 0.75→0.925) and `pubmed_k` 8→12 to match the labeler's evidence budget (conservatism halved); `alignment_accuracy` 0.325→0.45, citation_precision 1.0, 0 catastrophic flips. Streamlit model-load crash fixed (disable HF tqdm bars). |
| v2.1 | 2026-06-11 | **Phase 2 gap analysis implemented** (`phase-2` branch). `gap_analysis()` in pipeline.py: NCD-only hybrid retrieval + rerank (policy side) joined to PubMed evidence via **topical join** (abstracts filtered by `source_ncd_number == policy_number`, so evidence and policy describe the same intervention; empty join → "Insufficient Evidence" rather than unrelated abstracts). PubMed side: dense top-`pubmed_k`(8), no cross-encoder (bge-reranker not trained on clinical text), sorted newest-first. UI auto-routes policy vs gap queries via regex signal scoring (no mode toggle). Fixed PubMed date parser (ElementTree childless-element falsy bug left 99% of years blank); re-fetched → 2457 abstracts, 0% empty years. Gap eval: `generate_golden_gap.py` (independent `gemini-2.5-flash-lite` labeler reads raw NCD + abstracts — breaks circular self-grading; Groq question cache + rate-limit backoff) and `judge_gap.py` (retrieval-gated `alignment_accuracy`, `alignment_label_match`, `ncd_recall`, `pmid_recall`, `citation_precision`, faithfulness over policy+pubmed). Interaction/feedback logging extended with `mode`, `alignment`, and PubMed sources. |
| v2.3 | 2026-06-17 | **Primary-evidence re-ingest + whole-NCD gap context + full 277-record eval.** (1) Re-fetched PubMed with a two-pass primary-evidence filter (RCT/meta-analysis/systematic-review/cohort first, unrestricted backfill) + MeSH study-type fallback → **3,219 abstracts / 296 topics**, ~89% primary evidence (was ~75% review/background). (2) Tier-aware grading (T1–T5 + background/unspecified) threaded through `fetch_pubmed.evidence_tier`, the labeler, and `_GAP_SYSTEM`. (3) Re-labeled + Claude-adjudicated golden set → **277 records** (excluded 280.2 white-cane & 80.7 refractive-keratoplasty as non-medical/statutory; 18 v2 adjudications fixing over-called Coverage-Gap/Overcoverage). (4) Added `pmid_recall_retrieved` (citation recall over *retrieved* reference PMIDs — isolates citation behavior from the labeler-vs-pipeline retrieval-mechanism mismatch; 0.637→0.740 on the same answers) + seeded-random faithfulness subsampling. (5) Sharpened the Partial-vs-Aligned boundary (positive test: policy has explicit eligibility criteria AND evidence supports an excluded-but-eligible population); **reverted** an Insufficient-gate loosening that net-regressed on a 125-record check. (6) **Whole-NCD gap context** — chunk retrieval now only *identifies* the governing NCD via a score-weighted vote (sigmoid-of-logit; runner-up added only if within 70% of top; **capped at 2**), then feeds the full NCD text and scopes evidence to it. Fixes Partial gaps lost when the criteria chunk fell outside the top-5 (e.g. 240.4 CPAP Aligned→Partial). NCD-selection backtest (n=277): primary top-1 **0.830**, expected-in-selected **0.892**. A heuristic CED/admin boilerplate filter was prototyped and **rejected** (reliably regressed CPAP). |
| v2.4 | 2026-06-25 | **Whole-NCD gap eval completed**. Full 277-record re-run under whole-NCD context (faithfulness on 22 random samples) vs old top-5-chunk baseline: `alignment_accuracy` 0.570→**0.542**, `alignment_kappa` 0.346→**0.433**, `faithfulness` 0.597→**0.705**, `ncd_recall` 0.942→**0.888**, **Partial** label_match 0.20→**0.33**. Verdict: a *trade, not a win* — accuracy regressed but the loss is **entirely** the capped NCD-selection's `ncd_recall` drop, not reasoning (which improved on kappa/faithfulness/Partial). Recommendation logged: keep full-text policy context, loosen the NCD-selection cap/`second_frac` to recover `ncd_recall` — *superseded in v2.5*. See §15.2.2. Also: re-ingest DB committed + orphan segments pruned; doc fix — labeler is `gemini-2.5-flash` (not flash-lite). |
| v3.0 | 2026-06-27 | **Phase 3 kickoff — dynamic LCD lookup as agentic tool use** (branch `phase-3`). See §16. LCDs resolved at query time (NCD-governs? → resolve MAC for the beneficiary's state → fetch the live LCD → classify); routing decisions deterministic, orchestration agentic. Scoped to 5 MACs (Noridian/CGS/WPS/Palmetto/NGS). Spike: CMS LCD list endpoint ignores all server-side filters (~969 LCDs, filter by contractor locally + on-demand detail fetch). Step 1 shipped: `src/lcd/jurisdiction.py` (state→MAC, `extract_state`, `resolve_mac`, `ncd_disposition`); `data/lcd_raw.json` pruned 500→487. |
| v2.5 | 2026-06-27 | **Selection-tuning ruled out; disambiguation step is the lever.** Two LLM-free backtests (cap held at 2): (a) `second_frac` sweep — recall ceiling **0.924** at frac=0 (< old 0.942; ~1.8pts of expected NCDs never enter the top-2 rank), recall↔contamination locked ~1:8, and 0.924×0.61 ≈ 0.564 < 0.570 by arithmetic; (b) scope-aware selection (admit runner-up iff hierarchically related) — of 26 runner-up=expected cases only **3 are related / 23 unrelated**, so scope-only cuts `multi%` 26→6.5 but *regresses* recall 0.888→0.841. Neither threshold nor structure decouples recall from contamination — the signal is **semantic, not numeric**. Next lever: a narrow upstream disambiguation step (one bounded LLM call: which NCD governs?). Added `scripts/sweep_second_frac.py` (cap-2) + `scripts/backtest_scope_select.py`. |

---

<details>
<summary><strong>Evaluation Results History</strong> — Phase 1 RAGAS runs (click to expand)</summary>

NCD subset only (119/198 LCD entries excluded — Phase 1). Append a row after each `python -m src.evaluation.judge` run.

| Date | Eval set | k | Threshold | Reranker top_n | Faithfulness | Answer Relevancy | Context Precision | Empty Retrieval | Citation Acc. | Policy Recall | Notes |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-------|
| 2026-06-08 | 19 NCD | 5 | — | — | 0.867 | NaN | 0.717 | — | — | — | First run. AnswerRelevancy NaN — embeddings 404 + Gemini n=1 bug. Fixed in v1.3. |
| 2026-06-08 23:02 | 79 NCD | 5 | 0.70 | — | **0.923** ✅ | 0.802 | 0.784 | 0.0% | 0.135† | — | True NCD-only baseline. Faithfulness target met. |
| 2026-06-09 03:51 | 79 NCD | 3 | 0.75 | — | 0.792 | 0.660 | 0.646 | 26.6% | 0.709 | — | Aggressive filtering backfired — empty retrieval too high. Reverted. |
| 2026-06-09 05:10 | 79 NCD | 5 | 0.70 | 3 | 0.791 | 0.762 | 0.789 | 10.1% | 0.835 | — | Reranker top_n=3 hurt faithfulness (-0.13). Changing to top_n=5 (reorder only). |
| 2026-06-09 19:02 | 79 NCD | 5 | 0.70 | 5 | 0.860 | 0.761 | **0.832** ✅ | 7.6% | 0.899 | — | HTML strip fix + title prepend + synonym expansion. Context Precision target met. Faithfulness still below no-reranker baseline — reranker remains suspect. |
| 2026-06-09 23:42 | 79 NCD | 10 | 0.65 | 5 | 0.782 | 0.827 | 0.921 | 0.0% | 0.949 | — | ⚠️ CONTAMINATED — index had 8473 chunks (4× duplicates from append-on-rebuild). Judge upgraded to flash. Numbers not comparable to prior rows. |
| 2026-06-10 00:57 | 78 NCD‡ | 10 | 0.65 | 5 | 0.821 | **0.857** ✅ | **0.892** ✅ | 1.3% | 0.886 | — | **First clean baseline** — deduplicated index (1983 chunks), double-escape entity fix, LCD addendum top-doc fix. Answer Relevancy target met. |
| 2026-06-10 | 79 NCD | 10 | 0.65 | 5 | 0.854 | 0.833 | **0.917** ✅ | **0.0%** ✅ | 0.911 | **0.987** ✅ | Hybrid BM25+dense RRF retrieval. AR dipped slightly (-0.024); all other metrics improved. |

† Citation accuracy 0.135 is a measurement artifact — old cache entries lack `source_policy_numbers`; regex fallback understates true rate.
‡ 1 additional empty retrieval vs prior runs (78 RAGAS samples instead of 79).

**Targets:** Faithfulness > 0.90 · Answer Relevancy > 0.85 ✅ · Context Precision > 0.80 ✅ · Empty Retrieval < 15% ✅ · Citation Acc. > 95% · Policy Recall > 90% ✅

</details>

---

<details>
<summary><h1>PHASE 1 — Coverage Policy Intelligence Agent</h1></summary>

---

## 1. Problem & Opportunity

The key user is **provider-side prior-authorization and denial-prevention staff** — PA coordinators, utilization-review nurses, and revenue-cycle / denial-management teams at hospitals and practices — who must confirm a planned service meets Medicare coverage criteria *before* submitting the prior-auth request or claim. When they miss a criterion, the result is a denial: rework, delayed patient care, appeals, and lost revenue. Preventing that denial at submission time is the job to be done.

Today they spend 15–30 min per query manually searching the CMS Medicare Coverage Database, which requires exact CMS terminology (not clinical terminology) and buries coverage logic in dense 5,000–10,000 word documents. A wrong answer — stating covered when policy says not covered — drives exactly the denial they are trying to prevent, so every output must be grounded and cited.

CMS-0057-F (2024) mandated increased prior-authorization transparency and faster decisions, raising the value of getting coverage right at submission time.

---

## 2. Users

| User | Job to be done | Pain |
|---|---|---|
| **Provider-side PA / denial-prevention staff (KEY persona)** | Confirm a planned service meets Medicare coverage criteria *before* submitting the PA request/claim, to prevent denials | Manual CMS search, CMS ≠ clinical terminology, criteria buried in long documents — a missed criterion becomes a denial, rework, and an appeal |
| Provider utilization-review / PA nurses | Determine whether a service meets medical necessity criteria before submission | Same search burden under throughput pressure across many cases |
| Denial-management / appeals staff | Find the governing policy + evidence to overturn a denial within the appeal window | Manual PubMed + CMS search under a 30–72 hour clock |
| Medical policy analysts | Monitor policy currency against evidence | No systematic tool to track gaps between internal policy, CMS, and evidence |

> Denial-prevention staff have zero tolerance for AI overconfidence. A hallucinated "covered" that gets acted on causes the exact denial the tool is meant to prevent — worse than no tool at all.

---

## 3. AI Architecture — Phase 1

Phase 1 uses RAG + prompt engineering — no agent loop. Coverage policy lookup is retrieval and synthesis, not autonomous multi-step reasoning.

### 3.1 Pipeline

| Step | Component | Details |
|---|---|---|
| 1. Input validation | Rule-based filter | Block PHI. Flag patient-specific queries. |
| 2. Query embedding | `BAAI/bge-small-en-v1.5` (local CPU) | Same model used at index and query time. Zero API cost. |
| 3. Hybrid retrieval | BM25 + ChromaDB + RRF | BM25 sparse (k=10) + dense vector search (k=10, threshold=0.65) fused via Reciprocal Rank Fusion (k=60). BM25 handles exact term/numeric matches; dense handles semantic similarity. Up to 20 merged candidates passed to reranker. |
| 4. Cross-encoder reranking | `BAAI/bge-reranker-base` (local CPU) | Scores each (query, chunk) pair jointly. Selects top 5 of merged candidates — real filtering, not just reordering. |
| 5. Context injection | Prompt template | Top-5 reranked chunks with NCD/LCD headers. LCD jurisdiction note injected only when the top-ranked doc is an LCD. |
| 6. Answer generation | `gemini-2.5-flash` — temperature=0 | Indefinite retry loop reads `retryDelay` from API error. Returns answer + source Documents. |
| 7. RAGAS evaluation | `gemini-2.5-flash` + `BAAI/bge-small-en-v1.5` | Faithfulness, AnswerRelevancy (bypass_n=True), LLMContextPrecisionWithoutReference, policy_recall, citation_accuracy. Per-sample scores → `logs/eval_samples_latest.json`. Cache path derived from PIPELINE_CONFIG — each config gets its own file. |
| 8. Response delivery | Streamlit | Chat interface with source expander. All interactions logged to `logs/interactions.jsonl`. |

> **LCD jurisdiction — current state:** Geographic awareness is a prompt instruction only — LCD responses include a jurisdiction warning when an LCD is retrieved. Full implementation (MAC region UI, jurisdiction filtering, LCD eval) is deferred to Phase 2.

---

## 4. Quality Metrics & Targets

| Metric | How measured | Target | Current |
|---|---|---|---|
| Faithfulness | RAGAS Faithfulness | > 90% | 0.854 |
| Answer Relevancy | RAGAS AnswerRelevancy | > 85% | 0.833 |
| Context Precision | RAGAS LLMContextPrecisionWithoutReference | > 80% | **0.917 ✅** |
| Citation accuracy | Policy numbers from retrieved docs appear in response | > 95% | 0.911 |
| Empty retrieval rate | % queries returning no chunks | < 15% | **0.0% ✅** |
| Policy recall | Expected NCD policy number present in retrieved chunks | > 90% | **0.987 ✅** |
| False coverage rate | % responses incorrectly stating "covered" — manually audited | **0% — critical** | Pending audit |
| Response time p95 | End-to-end latency | < 8s | — |
| Cost per query | Embedding + reranking + generation + judge | < $0.06 | ~$0.004 |

---

## 5. Evaluation Framework

### 5.1 Golden dataset

198 Q&A pairs in `data/golden_dataset.json`. Fields: `question`, `reference_answer`, `policy_number`, `title`, `source` ("NCD"/"LCD"), `requires_jurisdiction` (bool).

- **79 NCD entries** — active eval set for Phase 1
- **119 LCD entries** — excluded from eval until Phase 2 jurisdiction handling; eval log reports coverage gap %
- Generated by `llama-3.1-8b-instant` (Groq) reading 3,000 chars of each policy. Target: 500 pairs by Month 3.

### 5.2 Evaluation pipeline (`judge.py`)

1. Load golden dataset → filter `requires_jurisdiction=true` → log coverage gap
2. Load answer cache (`logs/rag_answers_cache_*.json`) → skip cached questions
3. Call `rag_answer()` for uncached NCD questions (1s delay; indefinite retry on rate limits)
4. RAGAS per-sample with 1s between calls; up to 10 retries per sample
5. Per-sample scores → `logs/eval_samples_latest.json`
6. Aggregate scores + timestamp → `logs/eval_results.jsonl`

---

## 6. Data Strategy

**Phase 1 sources (all public):**
- NCDs: ~400 documents via CMS Coverage API
- LCDs: ~2,000+ documents via CMS Coverage API
- Refresh: weekly diff-check, re-index updated documents within 48h

**Data flywheel signals:**

| Signal | Meaning | Action |
|---|---|---|
| Reviewer verifies via cms.gov link | Answer trusted enough to verify | Log query + response |
| Reviewer triggers escalation | Output not trusted | Log as correction candidate |
| Follow-up question immediately | Answer incomplete | Follow-up + context = new golden pair |
| Session abandoned < 15s | Answer wrong/irrelevant | Log as negative training candidate |

---

## 7. Human-in-the-Loop

- **Faithfulness gate:** Score below threshold → route to human reviewer
- **Empty retrieval:** No policy found → structured "no policy found" + route to medical policy team
- **Reviewer corrections:** Logged and tagged for golden dataset expansion and fine-tuning (Phase 4)

---

## 8. Failure Modes

| Failure mode | Mitigation |
|---|---|
| Wrong NCD/LCD retrieved | Display full title + ID before synthesis. Reviewer verifies source. |
| Outdated policy cited | Weekly diff-check. Surface policy version date prominently. |
| Conditional coverage missed ("covered if A and B" → "covered") | Structured output with explicit conditional criteria field. Judge evaluates condition preservation. |
| Vocabulary gap (clinical term ≠ CMS term) | Title prepended to every chunk at index time (v1.6). MLN article ingestion as deeper vocabulary bridge (Month 2). |
| Geographic variation missed | LCD jurisdiction prompt note (v1.2). Full MAC region UI enforcement → Phase 2. |
| PHI in query | Input validation blocks PHI before any LLM call or storage. |
| Prompt injection via policy document | Document sanitisation at ingestion. Output validation before delivery. |

---

## 9. UX & Compliance

**UX principles:**
- Show retrieved NCD/LCD source before synthesised answer
- "No matching policy found" is a valid, correct output
- Every response includes: *"For informational support only. Coverage determinations require clinical judgment and qualified professional review."*
- MAC region visually prominent for LCD results

**Compliance:**
- No PHI sent to any external API
- CMS-0057-F (2024): outputs are documentable and auditable via `logs/interactions.jsonl`
- All data sources are public CMS data — no HIPAA data in pipeline

---

## 10. Cost Model

| Step | Model | Cost/query |
|---|---|---|
| Embedding + reranking | Local CPU | $0.00 |
| Answer generation | `gemini-2.5-flash` (~8K input, ~400 output tokens) | ~$0.003 |
| RAGAS judge | `gemini-2.5-flash-lite` (~9K input, ~200 output tokens) | ~$0.001 |
| **Total** | | **~$0.004** |

---

## 11. Phase 1 Roadmap

| Timeline | Target |
|---|---|
| Week 1 | NCD ingestion + indexing. Basic RAG chain. |
| Week 2 | LCD ingestion. Streamlit UI. Interaction logging. |
| Week 3 | RAGAS pipeline. Golden dataset (198 pairs). Faithfulness > 90% ✅ |
| Week 4 | Reranker. Rate-limit hardening. Persistent cache. LCD prompt instruction. |
| Month 2 | Expert review of 20 golden pairs/month. Vocab bridge (MLN articles). Answer Relevancy > 85%. |
| Month 3 | Golden dataset → 500 pairs. Citation link-checking. False coverage rate audit. |

</details>

---

<details open>
<summary><h1>PHASE 2 — Evidence vs Coverage Gap Analyzer</h1></summary>

---

## 12. Overview

Phase 1 answers: *"What does CMS cover for this?"*
Phase 2 answers: *"Where is CMS coverage out of step with published clinical evidence?"*

### 12.1 Persona — same workflow, back door instead of front door

The aligned user is **provider-side denial-management / appeals specialists** — specifically those handling **medical-necessity** and **experimental / investigational** denials. This is not a second audience; it is the *other end of the same revenue-cycle workflow* Phase 1 serves:

- **Phase 1 sits at the front of the revenue cycle** — pre-service: *prevent* the denial by getting coverage criteria and documentation right before submitting.
- **Phase 2 sits at the back** — the service was denied anyway, usually as *"not medically necessary"* or *"experimental/investigational,"* and someone now has to build the appeal.

That appeal is *exactly* an evidence-vs-coverage argument: the NCD/LCD doesn't cover this (or covers it too narrowly), **but here is the current literature supporting medical necessity for this patient.** That is precisely the structured gap report Phase 2 produces — the `Coverage Gap` and `Partial Coverage Gap` labels, with cited PMIDs and evidence grade, are the backbone of the appeal letter.

Same provider organization, same RCM / UM team, often the same patient — **prevention at the front door (Agent 1), recovery at the back door (Agent 2).** Medical policy teams (payer-side) also pay $200K–500K per engagement for periodic manual PubMed-vs-CMS reviews; Phase 2 automates that continuously, but the primary aligned persona is the provider appeals specialist. Introduced after Phase 1 reaches stable quality (Faithfulness > 90% ✅, False Coverage Rate = 0%).

---

## 13. Phase 2 Architecture — Evidence Gap Analysis (implemented)

`gap_analysis()` in `src/rag/pipeline.py`. Both retrieval legs are local (embeddings + reranker on CPU); only generation calls an API.

| Step | Component | Details |
|---|---|---|
| 1. Routing | `_route()` in `app.py` | Regex signal scoring routes each query to Policy Q&A or Gap Analysis — no manual mode toggle. Evidence words ("evidence", "studies", "RCT") vs policy words ("covered", "criteria", "NCD"). |
| 2. Policy identification + whole-NCD context | `_hybrid_retrieve_ncd` + `_rerank_scored` + `_select_primary_ncds` + `_full_ncd_docs` | **NCD-only** hybrid BM25+dense (LCDs excluded), RRF-fused, cross-encoder reranked. The reranked chunks only *identify* the governing NCD via a **score-weighted vote** (sigmoid of the cross-encoder logit; a runner-up NCD is added only if within 70% of the top; selection **capped at 2** so an ambiguous query can't pull 3–5 full policies). The **full text of the primary NCD(s)** is then supplied as context — not the top-5 question-similar chunks — so the eligibility-criteria section (which distinguishes a Partial Coverage Gap from Aligned) can't be dropped by chunk ranking (v2.3; e.g. 240.4 CPAP). |
| 3. Topical join | `_pubmed_for_ncds` | PubMed abstracts pulled **only for the primary NCD(s)** (`source_ncd_number == policy_number`), so evidence and coverage position describe the same intervention. Dense top-`pubmed_k`(12), no threshold (the NCD filter is the topicality gate), no cross-encoder (bge-reranker-base is not trained on clinical abstracts), sorted newest-first. **Empty join → "Insufficient Evidence"** rather than unrelated abstracts from an open search. |
| 4. Gap synthesis | `gemini-2.5-flash` — temperature=0 | Structured report: **CMS Coverage Position · Clinical Evidence (one `PMID <id>` bullet per abstract) · Evidence Grade · Alignment · Gap Summary**. Prompt forbids citing PMIDs not in the provided abstracts; pins Evidence Grade/Alignment to "Insufficient" when no abstracts are retrieved. |
| 5. Delivery | Streamlit | Separate "CMS Policy Sources" and "PubMed Evidence" expanders (PMID · year · journal · excerpt). Interactions logged with `mode`, parsed `alignment`, and both source sets. |

**Alignment taxonomy (action-oriented, by direction of divergence):** Aligned (CMS & evidence agree — no action) · Partial Coverage Gap (covers, but narrower than evidence supports — broaden) · Coverage Gap (evidence supports, CMS doesn't cover/denies — expand/appeal) · Overcoverage (CMS covers, evidence weak/absent/negative — utilization review) · Insufficient Evidence (can't judge — manual review). Each label maps to one analyst action; there is deliberately no "conflicting" label — a divergence is classified by *which side is ahead*, since that determines the action.

**Full LCD jurisdiction implementation (still deferred):** MAC region UI, reviewer jurisdiction profile, LCD re-admission to eval, MAC-region retrieval filtering — not yet built.

---

## 14. Phase 2 Data Sources

| Source | Coverage | Access | Status |
|---|---|---|---|
| CMS NCDs/LCDs | 1,983 chunks (`cms_coverage`) | CMS Coverage API | Indexed |
| PubMed/MEDLINE | **3,219 abstracts** (`pubmed_evidence`) across 296 NCD topics, ≤12 per topic, deduped by PMID; ~89% primary evidence | NCBI E-utilities (free, 3 req/s) | **Ingested (v2.3 re-fetch)** — `fetch_pubmed.py` runs a two-pass per-NCD search (primary-evidence filter: RCT/meta-analysis/systematic-review/cohort first, then unrestricted backfill), captures NLM PublicationType + a MeSH study-type fallback for tier grading, and tags each abstract with `source_ncd_number` for the topical join |
| ClinicalTrials.gov | ~500K trials | ClinicalTrials API v2 | Planned |

> **PubMed date parsing:** the original parser hit the ElementTree gotcha where a childless `<Year>` element is falsy, so `find(Year) or find(MedlineDate)` skipped real years — 99% of abstracts had blank years. Fixed with explicit `None` checks + a MedlineDate year-regex fallback; re-fetch yields 0% empty years (enables newest-first evidence ordering).

---

## 15. Phase 2 Evaluation, Metrics & Roadmap

### 15.1 Gap evaluation framework

Breaking the circularity: an early version graded the pipeline against labels produced by *running the pipeline itself* — measuring reproducibility, not correctness. The reference labels are now produced by an **independent judge**.

- **`generate_golden_gap.py`** — for each NCD with abstracts: (1) Groq `llama-3.1-8b-instant` writes one evidence-seeking question (cached in `data/gap_questions.json`, rate-limit backoff); (2) an **independent `gemini-2.5-flash` labeler** reads the *raw* NCD text + that NCD's abstracts (never the pipeline's report) and emits `reference_alignment`, `reference_pmids`, rationale → `data/golden_gap.json`. Different prompt + raw evidence + cross-vendor Claude adjudication = labels independent of the pipeline's generation. (Upgraded from flash-lite, which could not reliably resolve name-collisions in the topical join.)
- **`judge_gap.py`** — runs `gap_analysis()` per question (answers cached by config) and scores it against the independent reference.

### 15.2 Metrics

| Metric | Measures | Target |
|---|---|---|
| `alignment_accuracy` | **End-to-end:** reached the reference alignment **AND** retrieved the right NCD (a correct label on the wrong retrieved policy = miss) | > 75% |
| `alignment_label_match` | Diagnostic: raw label agreement vs reference, retrieval-blind — isolates reasoning from retrieval | — |
| `alignment_action_match` | Provider action-bucket match — appeal `{Partial, Coverage Gap}` / covered `{Aligned, Overcoverage}` / manual `{Insufficient}` | — |
| `alignment_kappa` | Quadratic-weighted Cohen's kappa over the ordinal evidence-vs-coverage direction (excludes Insufficient) | — |
| `ncd_recall` | Expected NCD surfaced into `policy_sources` by retrieval (not parroted from context) | > 90% |
| `pmid_recall` | Fraction of the reference's key PMIDs the report cited | > 60% |
| `pmid_recall_retrieved` | Citation recall over reference PMIDs that were **actually retrieved** — isolates citation behavior from the labeler-vs-pipeline retrieval-mechanism mismatch | — |
| `citation_precision` | Fraction of cited PMIDs that were actually retrieved (catches fabricated citations) | > 95% |
| `faithfulness` | RAGAS faithfulness of the gap report vs policy + PubMed contexts | > 90% |

`alignment_accuracy = label_match ∧ ncd_recall`, so the three decompose failures: high `label_match` + low `ncd_recall` ⇒ retrieval is the bottleneck; the reverse ⇒ reasoning is.

**Reference set:** 284 records, each labeled by the flash-lite labeler then **fully hand-adjudicated by Claude** (cross-vendor) against the raw NCD + abstracts — 67 overrides (24%), 217 confirmed. Adjudication surfaced (and fixed) a name-collision bug: the topical join retrieves abstracts fetched by the NCD *title*, so ambiguous titles pull a different same-named intervention (CAR-T for "Cellular Therapy", sacral neuromod for "Bladder Stimulators"). Fixed via a disambiguation step in `_GAP_SYSTEM`/`_LABEL_PROMPT` (flash resolves these to Insufficient; flash-lite cannot, so `LABEL_MODEL` upgraded flash-lite→flash).

### 15.2.1 Gap-eval results (40-sample, 2026-06-13)

The first run surfaced issues; three legitimate fixes were applied and re-run (same 40 NCDs):

| Metric | v1: k=8, evidence-meta Q | v2: k=12, topic-forward Q | v3: + ported rubric |
|---|:---:|:---:|:---:|
| `alignment_accuracy` (end-to-end) | 0.325 | 0.45 | **0.525** |
| `alignment_label_match` | 0.40 | 0.475 | **0.525** |
| `alignment_action_match` (provider buckets) | — | 0.525 | **0.525** |
| `alignment_kappa` (quadratic-weighted) | — | 0.349 | **0.424** |
| `ncd_recall` | 0.75 | 0.925 | **0.925** |
| `pmid_recall` | 0.57 | 0.80 | **0.844** |
| `citation_precision` | **1.00** | **1.00** | **1.00** |
| `faithfulness` | 0.70 (n=10) | 0.78 (n=20) | 0.76 (n=20) |

**0 catastrophic (Aligned↔Coverage-Gap) flips in any run.** `alignment_accuracy` rose 0.325→0.525 (+62%) with **no reference dumbing-down** — just better questions, enough evidence, and the same rubric the gold standard uses.

The three fixes, with clean attribution:
1. **Topic-forward questions** (`ncd_recall` 0.75→0.925, *purely* the question fix since `ncd_recall` is policy-side). The v1 synthetic questions were drenched in evidence-meta vocabulary ("RCTs, observational studies, improved outcomes"), biasing retrieval toward trial-heavy *Coverage-with-Evidence-Development* NCDs (TAVR/TEER/warfarin-PGx); topic-forward queries (lead with the intervention) retrieve the right NCD 37/40.
2. **`pubmed_k` 8→12** (`Partial/Gap→Insufficient` conservatism halved 6→3). The labeler judges from up to 12 abstracts deliberately (holistic, in lieu of human review); k=8 under-surfaced the answer the fuller evidence supports. Matching the pipeline's evidence budget to the reference's confirmed the reference is the gold standard and k=8 was the limiter, not an unfair handicap.
3. **Ported the labeler's decision rubric into `_GAP_SYSTEM`** (`label_match` 0.475→0.525, rising to *equal* `action_match`). Before this, the pipeline often picked the right *action bucket* but the wrong *sub-label* (Partial vs full Coverage Gap); giving it the same ordered procedure + "covers-any-indication→never-a-full-gap" rule eliminated the within-bucket disagreements. Residual misses are now genuinely *cross-action* (appeal vs covered vs manual), not label hairsplitting.

**Provider-facing metric:** `alignment_action_match` collapses labels to the appeals-specialist action — appeal `{Partial, Coverage Gap}` / covered `{Aligned, Overcoverage}` / manual `{Insufficient}` — measuring whether the tool gets the *decision* right, not the exact severity label.

**Known limitations:** strict 5-class exact match reads pessimistically on a subjective task; reference labels are LLM-drafted + Claude-adjudicated (not clinician-validated). Next lever: spot-check residual mismatches to separate pipeline-error from reference-error (some borderline adjudications are likely the "miss").

### 15.2.2 Full 277-record runs: top-5-chunk baseline → whole-NCD redesign (v2.3–v2.4)

**Baseline (top-5-chunk, 2026-06-17, v2.3).** Moved from a 40-sample slice to the **full 277-record** reference set (after re-ingest, re-label, and removal of the two non-medical NCDs); faithfulness on 15 random samples to cap cost. Baseline numbers are the left column of the consolidated table below.

Per-label diagnosis (label_match): Aligned 0.71, Insufficient 0.66, Coverage Gap 0.67, Overcoverage 0.30, **Partial 0.20** — Partial collapsing to Aligned (30/55, 28 with the right NCD retrieved) was the dominant error and a *reasoning* miss, not retrieval. `pmid_recall_retrieved` (0.740) vs `pmid_recall` (0.637) confirmed ~10 pts of the apparent citation "loss" was the retrieval-mechanism mismatch, not the pipeline (precision 0.998 = zero fabrication).

**Retrieval redesign (whole-NCD context).** Root cause of the Partial collapse: chunk-level top-5 retrieval can omit the eligibility-criteria section that defines a Partial gap (240.4 CPAP — the AHI/comorbidity criteria chunk never made the top-5 → Aligned). Fix: identify the governing NCD (score-weighted vote, capped at 2) and feed its full text + scope evidence to it. Spot-checks: 240.4 CPAP Aligned→**Partial** ✓, 100.1 bariatric stays **Aligned** ✓, 260.1 liver still misses (a genuine Partial-vs-Aligned *reasoning* boundary issue, not retrieval — its supporting RCT was confirmed present in the scoped evidence). NCD-selection backtest (n=277, LLM-free): primary top-1 **0.830**, expected-in-selected **0.892** (vs the old "all NCDs in top-5 chunks" 0.942 — the dip is mostly sibling/parent-child NCDs and a deliberate contamination/recall trade; `second_frac=0.7` sits at the knee of the sweep).

**Rejected:** a heuristic CED/admin boilerplate filter (drop research-protocol chunks from the whole-NCD context) — it reliably regressed CPAP Partial→Aligned across 4 runs despite retaining the criteria chunk, so it was reverted. Verdicts are context-composition-sensitive; any content trimming must be section-aware and eval-validated, not heuristic.

**Whole-NCD eval (2026-06-25, v2.4).** The deferred re-run is done — full 277 records regenerated under the whole-NCD pipeline (faithfulness on 22 random samples). Head-to-head vs the top-5-chunk baseline above:

| Metric | Old top-5-chunk | Whole-NCD | Δ |
|---|:---:|:---:|:---:|
| `alignment_accuracy` | **0.570** | 0.542 | 🔴 −0.028 |
| `alignment_label_match` | **0.581** | 0.570 | 🔴 −0.011 |
| `alignment_action_match` | 0.621 | 0.621 | ⚪ 0.000 |
| `alignment_kappa` | 0.346 | **0.433** | 🟢 +0.087 |
| `ncd_recall` | **0.942** | 0.888 | 🔴 −0.054 |
| `pmid_recall` | 0.637 | 0.633 | ⚪ −0.004 |
| `pmid_recall_retrieved` | 0.740 | 0.725 | ⚪ −0.015 |
| `citation_precision` | 0.998 | 0.994 | ⚪ −0.004 |
| `faithfulness` (random) | 0.597 (n=15) | **0.705** (n=22) | 🟢 +0.108 |

Per-label `label_match` (old→new): Aligned 0.71→0.60, Insufficient 0.66→0.70, Coverage Gap 0.67→**0.83**, Overcoverage 0.30→0.30, **Partial 0.20→0.33**.

**Verdict — a trade, not a win.** Headline `alignment_accuracy` *regressed* (0.570→0.542), but the drop is **entirely a retrieval regression, not reasoning**: since `accuracy = label_match ∧ ncd_recall` and label_match held, the loss tracks `ncd_recall` (0.942→0.888) — the score-weighted **capped-at-2 NCD selection** surfaces the governing NCD less often than the old "any NCD in the top-5 chunks." Reasoning quality *improved*: kappa +0.087 (better evidence-vs-coverage *direction*), faithfulness +0.108 (full policy text → less fabrication), Coverage Gap +0.16, and Partial finally off the floor (0.20→0.33). But Partial's gain came at **Aligned's expense** (0.71→0.60, ≈+7 Partial / −14 Aligned) — the Partial/Aligned boundary *moved*, it didn't *sharpen*.

**Recommendation.** Whole-NCD bundled two changes: (1) full-text policy context — clearly good (kappa, faithfulness, Partial all up); (2) capped score-weighted NCD selection — clearly costly (`ncd_recall` −0.054). **Keep (1); the fix for (2) is *smarter* selection, not *looser*** — two LLM-free backtests (2026-06-27) close the parameter path:

- **`second_frac` sweep, cap held at 2** (`scripts/sweep_second_frac.py`): the recall ceiling is **0.924 at `frac=0`** — below the old 0.942, because ~1.8 pts of expected NCDs never enter the top-2 aggregate rank, so *no* threshold reaches them. And recall↔contamination are locked ~1:8 (frac 0.70→0.50 buys +2.5 pts recall for +18 pts `multi%`). Even the 0.924 ceiling × conditional-label-accuracy 0.61 ≈ **0.564 < 0.570** — arithmetic alone rules `second_frac` out.
- **Scope-aware selection** (`scripts/backtest_scope_select.py`, admit runner-up only if hierarchically related — shared dotted-prefix ≥2 components): of the 26 cases where the expected NCD is the runner-up, **only 3 are related; 23 are unrelated**. So scope-only cuts contamination (`multi%` 26→6.5) but *regresses* recall (0.888→0.841) — it rejects the correct-but-unrelated runner-ups, which are structurally indistinguishable from strays. **The discriminating signal is semantic, not numeric.**

Neither threshold nor structure decouples recall from contamination. The remaining lever is a **narrow disambiguation step**: when top-1 and a runner-up score close, one bounded LLM call reads both NCD headers + the question and decides which actually govern — picking the *right* policy, not merely *more* policies (so it can lift conditional label-accuracy, not just recall). It's the first place in the gap pipeline a reasoning loop is empirically justified: a single upstream "which NCD governs?" decision, not general agency. The Aligned↔Partial boundary remains a separate, focused problem.

### 15.3 Roadmap

| Milestone | Status |
|---|---|
| PubMed ingestion + indexing | ✅ Done (2,457 abstracts) |
| Topical-join retrieval + gap synthesis | ✅ Done |
| Independent gap eval (golden set + judge) | ✅ Done — full 277-record run complete under both top-5-chunk (v2.3) and whole-NCD (v2.4) pipelines; human validation pending |
| Full LCD jurisdiction implementation | 🔨 In progress — **Phase 3** (dynamic LCD lookup as tool use) |
| Expert validation (20 gap reports/month) | Planned |
| Fine-tuning on gap assessments | Planned |

</details>

---

## 16. Phase 3 — Dynamic LCD Lookup (agentic tool use)

**Thesis.** LCDs are jurisdictional and too numerous to pre-index sensibly (one CMS list call returns ~969 final LCDs, the *same* service has different LCDs across MACs, and the relevant one depends on the beneficiary's state). So coverage must be resolved at query time. This is the one place agentic orchestration earns its keep — a capability static RAG structurally cannot have.

**Control flow** (branch `phase-3`):

```
query (+ optional state)
  → NCD-resolver  ── governs ─▶ classify with NCD (reuse Phase-2 pipeline)   ✅
       │ (deterministic: disambiguation + defer-marker check)
       └─ silent / defers-to-MAC
            → resolve MAC for the state   (deterministic table; ask the user
            │   for their state if the query has none)
            → TOOL lcd_lookup(mac, service)   (dynamic CMS API call)
                 ├─ hit(s) → classify against the live LCD(s)
                 ├─ empty  → "no LCD → contractor discretion / individual consideration"
                 └─ error  → degrade gracefully, report (no fabrication)
```

**Deterministic vs agentic boundary (by design).** Every routing *decision* stays deterministic — `ncd_disposition` (governs/defers/silent via a high-precision defer-marker regex), `extract_state`, `resolve_mac`. The **agentic** part is orchestrating *when* to call which tool, plus reconciling multiple LCDs and handling empty/error/fallback.

**Scope.** Five MAC jurisdictions end-to-end — **Noridian, CGS, WPS, Palmetto, National Government Services** (the best-represented; 487/500 snapshot LCDs). States served by First Coast and Novitas are out of scope and resolve to *unsupported jurisdiction* (reported, not guessed).

**Spike finding (CMS Coverage API).** The LCD list endpoint `reports/local-coverage-final-lcds` **ignores all server-side filters** (keyword/state/contractor/q/search are no-ops) and returns ~969 LCDs in one call; no by-state/search endpoint exists. LCDs carry no state field, only `contractor_name_type`. So `lcd_lookup` = fetch the list once (cacheable) → filter by contractor locally → fetch full text on demand via the license-token-gated detail endpoint.

**Status.** *Step 1 done* — `src/lcd/jurisdiction.py` (state→MAC table for the 5 MACs, `extract_state`, `resolve_mac`, `ncd_disposition`; 15/344 NCDs detected as genuine MAC hand-offs). `data/lcd_raw.json` pruned 500→487. *Next:* (2) `lcd_lookup` tool against the live API with empty/error fallback; (3) the Gemini function-calling orchestration loop; (4) a jurisdictional eval set (service × state → expected LCD/disposition). The `cms_coverage` index still holds the old LCD set — rebuild when the dynamic path lands.

---

*Medicare Coverage Intelligence Platform · PRD v3.0 (Phase 3 in progress) · All data sources public · Last updated 2026-06-27*
