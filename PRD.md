# Medicare Coverage Intelligence Platform — PRD

**Product:** Medicare Coverage Intelligence Platform
**Domain:** Health Insurance / Utilization Management
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

---

## Evaluation Results History

NCD subset only (119/198 LCD entries excluded — Phase 1). Append a row after each `python -m src.evaluation.judge` run.

| Date | Eval set | k | Threshold | Reranker top_n | Faithfulness | Answer Relevancy | Context Precision | Empty Retrieval | Citation Acc. | Notes |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-------|
| 2026-06-08 | 19 NCD | 5 | — | — | 0.867 | NaN | 0.717 | — | — | First run. AnswerRelevancy NaN — embeddings 404 + Gemini n=1 bug. Fixed in v1.3. |
| 2026-06-08 23:02 | 79 NCD | 5 | 0.70 | — | **0.923** ✅ | 0.802 | 0.784 | 0.0% | 0.135† | True NCD-only baseline. Faithfulness target met. |
| 2026-06-09 03:51 | 79 NCD | 3 | 0.75 | — | 0.792 | 0.660 | 0.646 | 26.6% | 0.709 | Aggressive filtering backfired — empty retrieval too high. Reverted. |
| 2026-06-09 05:10 | 79 NCD | 5 | 0.70 | 3 | 0.791 | 0.762 | 0.789 | 10.1% | 0.835 | Reranker top_n=3 hurt faithfulness (-0.13). Changing to top_n=5 (reorder only). |
| 2026-06-09 19:02 | 79 NCD | 5 | 0.70 | 5 | 0.860 | 0.761 | **0.832** ✅ | 7.6% | 0.899 | HTML strip fix + title prepend + synonym expansion. Context Precision target met. Faithfulness still below no-reranker baseline — reranker remains suspect. |
| 2026-06-09 23:42 | 79 NCD | 10 | 0.65 | 5 | 0.782 | 0.827 | 0.921 | 0.0% | 0.949 | ⚠️ CONTAMINATED — index had 8473 chunks (4× duplicates from append-on-rebuild). Judge upgraded to flash. Numbers not comparable to prior rows. |
| 2026-06-10 00:57 | 78 NCD‡ | 10 | 0.65 | 5 | 0.821 | **0.857** ✅ | **0.892** ✅ | 1.3% | 0.886 | **First clean baseline** — deduplicated index (1983 chunks), double-escape entity fix, LCD addendum top-doc fix. Answer Relevancy target met. |

† Citation accuracy 0.135 is a measurement artifact — old cache entries lack `source_policy_numbers`; regex fallback understates true rate.
‡ 1 additional empty retrieval vs prior runs (78 RAGAS samples instead of 79).

**Targets:** Faithfulness > 0.90 · Answer Relevancy > 0.85 ✅ · Context Precision > 0.80 ✅ · Empty Retrieval < 15% ✅ · Citation Acc. > 95% · Policy Recall > 90% ✅

---

---

# PHASE 1 — Coverage Policy Intelligence Agent

---

## 1. Problem & Opportunity

UM reviewers and medical policy teams spend 15–30 min manually searching CMS Medicare Coverage Database per query. The CMS interface requires knowing exact CMS terminology (not clinical terminology), and coverage logic is buried in dense 5,000–10,000 word documents. A wrong answer — stating covered when policy says not covered — can cause wrongful denials, audit findings, or member harm. Every output must be grounded and cited.

CMS-0057-F (2024) mandated increased PA transparency, giving plans that systematically document coverage policy review a compliance and legal defensibility advantage.

---

## 2. Users

| User | Job to be done | Pain |
|---|---|---|
| UM nurses / PA reviewers | Determine whether a service meets Medicare medical necessity criteria before a PA decision | Manual search, CMS ≠ clinical terminology, conditional logic buried in dense documents |
| Medical policy analysts | Monitor policy accuracy and currency against evidence | No systematic tool to track gaps between internal policy, CMS, and evidence |
| Appeals reviewers | Find policy + clinical evidence within 30–72 hour window | Manual PubMed + CMS search under time pressure |

> UM reviewers have zero tolerance for AI overconfidence. A hallucinated coverage determination that gets acted on is worse than no tool at all.

---

## 3. AI Architecture — Phase 1

Phase 1 uses RAG + prompt engineering — no agent loop. Coverage policy lookup is retrieval and synthesis, not autonomous multi-step reasoning.

### 3.1 Pipeline

| Step | Component | Details |
|---|---|---|
| 1. Input validation | Rule-based filter | Block PHI. Flag member-specific queries. |
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
| Faithfulness | RAGAS Faithfulness | > 90% | 0.821 |
| Answer Relevancy | RAGAS AnswerRelevancy | > 85% | **0.857 ✅** |
| Context Precision | RAGAS LLMContextPrecisionWithoutReference | > 80% | **0.892 ✅** |
| Citation accuracy | Policy numbers from retrieved docs appear in response | > 95% | 0.886 |
| Empty retrieval rate | % queries returning no chunks | < 15% | **1.3% ✅** |
| Policy recall | Expected NCD policy number present in retrieved chunks | > 90% | **0.962 ✅** |
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

---

---

# PHASE 2 — Evidence vs Coverage Gap Analyzer

---

## 12. Overview

Phase 1 answers: *"What does CMS cover for this?"*
Phase 2 answers: *"Where is CMS coverage out of step with published clinical evidence?"*

Medical policy teams currently pay $200K–500K per engagement for periodic manual PubMed-vs-CMS reviews. Phase 2 automates this continuously. Introduced after Phase 1 reaches stable quality (Faithfulness > 90% ✅, False Coverage Rate = 0%).

---

## 13. Phase 2 Architecture

**Multi-source agentic pipeline:**
1. RAG retrieval from CMS NCD/LCD corpus (Phase 1 pipeline, unchanged)
2. PubMed/MEDLINE search for clinical evidence on the same service/indication
3. Synthesis agent: compare coverage criteria vs evidence strength → identify gaps, conflicts, alignment
4. Structured gap report: coverage position · evidence grade · gap type · recommended action

**Full LCD jurisdiction implementation (deferred from Phase 1):**
- MAC region metadata surfaced in UI — visually prominent, not dismissible
- Reviewer jurisdiction profile — LCDs from other jurisdictions auto-flagged
- LCD entries re-admitted to eval with jurisdiction-aware scoring
- Retrieval filtering by reviewer's MAC region

---

## 14. Phase 2 Data Sources

| Source | Coverage | Access |
|---|---|---|
| CMS NCDs/LCDs | ~2,400+ documents | CMS Coverage API (already indexed) |
| PubMed/MEDLINE | 35M+ abstracts | NCBI E-utilities (free, rate-limited) |
| ClinicalTrials.gov | ~500K trials | ClinicalTrials API v2 (free) |

---

## 15. Phase 2 Metrics & Roadmap

| Metric | Target |
|---|---|
| Gap identification accuracy (analyst-confirmed) | > 80% |
| False positive gap rate | < 20% |
| Evidence grade accuracy (RCT / meta-analysis / observational) | > 90% |

| Milestone | Target |
|---|---|
| PubMed ingestion + indexing | Month 4 |
| Multi-source retrieval + gap synthesis agent | Month 5 |
| Full LCD jurisdiction implementation | Month 5 |
| Expert validation (20 gap reports/month) | Month 6+ |
| Fine-tuning on gap assessments | Month 7+ |

---

*Medicare Coverage Intelligence Platform · PRD v2.0 · All data sources public · Last updated 2026-06-10*
