# Medicare Coverage Intelligence Platform — Product Requirements Document

**Product:** Medicare Coverage Intelligence Platform
**Domain:** Health Insurance / Utilization Management
**Data sources:** CMS NCDs/LCDs (Phase 1) · PubMed (Phase 2)

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2026-03-01 | urvi367 | Initial PRD — Phase 1 architecture with Groq/llama + HuggingFace embeddings |
| v1.1 | 2026-05-01 | urvi367 | Updated Phase 1 architecture to reflect live implementation |
| v1.2 | 2026-06-07 | urvi367 | Switch answer generation to gemini-2.5-flash, judge to gemini-2.5-flash-lite; add Context Relevance eval metric; persistent answer cache; golden dataset gets document_type + requires_jurisdiction fields; LCD jurisdiction handled via prompt instruction (full implementation deferred to Phase 2); rate-limit retry using API retryDelay; LCD entries filtered from eval run until jurisdiction handling is complete |

---

---

# PHASE 1 — Coverage Policy Intelligence Agent

---

## 1. Problem & Opportunity

### 1.1 The core problem

UM reviewers and medical policy teams at health plans spend significant time manually searching the CMS Medicare Coverage Database before making prior authorization decisions or writing denial letters. The CMS search interface is poor, NCD and LCD documents are dense legal text, and locating the correct policy requires knowing the exact CMS terminology — not the clinical terminology a reviewer would use.

A second, related problem: Medicare coverage policies lag clinical evidence by 3–7 years. Plans default to existing CMS policy while strong evidence accumulates for newly indicated services. This creates a systematic gap that generates avoidable denials, drives appeals volume, and disadvantages members who need evidence-supported care.

### 1.2 Why this is an AI problem, not a search problem

| Question | Answer |
|----------|--------|
| Why can't this be solved with better search? | Coverage questions are not keyword lookup tasks. A reviewer asking "does Medicare cover SGLT-2 inhibitors for heart failure in a non-diabetic patient?" needs the system to reason about conditional coverage criteria, not return a list of links. Keyword search returns the right document 40% of the time at best; the reviewer still has to read 5,000 words to find the answer. |
| What data exists to ground the model? | ~400 NCDs and ~2,000+ LCDs are publicly available as structured XML from the CMS Coverage API (api.coverage.cms.gov/v1). Plus the Medicare Benefit Policy Manual and CMS MLN articles. All public, all downloadable, no licensing barrier. |
| What is the cost of a wrong answer? | HIGH. An incorrect coverage determination — stating covered when policy says not covered, or vice versa — can result in a wrongful denial, a regulatory audit finding, or member harm. Every output must be grounded and cited. |
| Why now? | CMS-0057-F (2024) mandated increased PA transparency. Plans that can demonstrate systematic, documented coverage policy review have a compliance and legal defensibility advantage. |

---

## 2. Users & Jobs to Be Done

### 2.1 Primary users

| User segment | Job to be done | Current pain |
|---|---|---|
| UM nurses / PA reviewers | Quickly determine whether a requested service meets Medicare medical necessity criteria before making a PA decision. | Manual search on cms.gov takes 15–30 min per complex query. CMS terminology differs from clinical terminology. Conditional coverage logic buried in dense documents. |
| Medical policy analysts | Monitor coverage policies for accuracy and currency against current evidence. | No systematic tool to track where internal policies are out of step with CMS or where CMS is out of step with evidence. |
| Appeals reviewers | Find clinical and policy evidence supporting or opposing a member appeal within a 30–72 hour window. | Manual PubMed + CMS search under time pressure. Inconsistent evidence quality across reviewers. |

### 2.2 AI-specific user considerations

- **Trust calibration:** Reviewers will initially distrust outputs. Design for verification-first UX — show the source NCD/LCD before showing the synthesised answer.
- **Error mode asymmetry:** Users are more accepting of "I could not find a definitive answer" than of a confident wrong answer. When confidence is low, say so explicitly.
- **Domain vocabulary mismatch:** Reviewers use clinical terms (empagliflozin, HFpEF, prior auth). CMS uses policy terms (HCPCS J0605, Medicare benefit category, coverage with evidence development). The retrieval system must bridge this vocabulary gap.

> **Watch out:** UM reviewers are high-stakes decision makers with low tolerance for AI overconfidence. A hallucinated coverage determination that gets acted on is worse than no tool at all.

---

## 3. AI Architecture — Phase 1

### 3.1 Architecture decision

Phase 1 uses RAG + prompt engineering. No agent loop required — coverage policy lookup is a retrieval and synthesis task, not a multi-step autonomous task. Agents are introduced in Phase 2.

| Approach | Decision | Reasoning |
|---|---|---|
| Prompt engineering only | Partial — used for output format and constraints | Cannot produce verifiable, policy-grounded answers alone. |
| RAG over CMS NCDs/LCDs | **YES — primary architecture** | CMS Coverage API used for live NCD and LCD fetching. 2,400+ documents indexed. RAG grounds every claim in a retrieved policy chunk. |
| Agentic workflow | Phase 2 only | Phase 1 does not require autonomous decision loops. |
| Fine-tuning | Phase 4 (Month 6+) | After 6 months of labeled reviewer corrections. |

### 3.2 Phase 1 pipeline — step by step

| Step | Component / model | Decision & threshold |
|---|---|---|
| 1. Input validation | Rule-based filter | Block queries containing PHI. Flag queries suggesting a specific member. |
| 2. Query embedding | `BAAI/bge-small-en-v1.5` (local, CPU) | Zero API cost. Used identically at index time and query time — consistency required for retrieval. |
| 3. NCD/LCD vector search | ChromaDB — persisted at `data/chroma/` | Top-5 chunks by cosine similarity. 800-char chunks / 100-char overlap. Do not generate from model memory if no relevant policy is retrieved. |
| 4. Evidence type filter | Metadata filter | Prefer NCDs over LCDs when both exist (NCDs are nationally binding; LCDs are regional). Surface MAC region for LCD results. |
| 5. Context injection | Prompt template | Inject top-5 retrieved chunks. System instruction: cite title and policy number for every claim. **LCD jurisdiction instruction (v1.2):** if any retrieved document is an LCD, state at the start of the answer: "Note: This determination is based on an LCD which applies to [jurisdiction] only. Coverage may differ in other MAC regions." If jurisdiction unknown, say so and instruct user to verify. If both NCD and LCD retrieved for same service, state NCD national position first, then how the LCD modifies criteria for the specific jurisdiction. |
| 6. Answer generation | `gemini-2.5-flash` (Google AI) | Temperature = 0. Retry loop reads `retryDelay` from API error and waits indefinitely on rate limits. Returns answer text + source Documents. |
| 7. Faithfulness judge | `gemini-2.5-flash-lite` (Google AI) via RAGAS | Metrics: Faithfulness, Answer Relevancy, Context Relevance (LLMContextPrecisionWithoutReference). Per-sample evaluation with 7s inter-call delay for free-tier rate limit. Persistent answer cache at `logs/rag_answers_cache.json` — NCD answers reused across runs; LCD entries filtered from eval until full jurisdiction handling is implemented. |
| 8. Response delivery | Streamlit | Chat interface with session history. Sources in expandable panel. All interactions logged to `logs/interactions.jsonl`. |

> **LCD jurisdiction — current state (v1.2):** Geographic awareness is implemented as a **prompt instruction only**. The prompt tells the model to surface jurisdiction warnings when it detects an LCD in the retrieved context. Full architectural support (MAC region metadata surfaced in UI, jurisdiction filtering by reviewer location, LCD-specific eval coverage) is deferred to Phase 2. LCD entries are **excluded from eval runs** until the jurisdiction evaluation framework is ready; the eval log reports the % of dataset excluded.

---

## 4. North Star Metrics & KPIs

> The primary quality signal is whether coverage determinations are correct and verifiable. A single wrong determination acted upon is a more serious failure than 100 sessions with poor engagement metrics.

### 4.1 AI quality metrics

| Metric | Definition | Threshold to ship |
|---|---|---|
| Faithfulness score | % of responses where every coverage determination is grounded in the retrieved NCD/LCD chunks. Scored by RAGAS Faithfulness. | > 90% |
| Citation accuracy | % of cited NCD/LCD IDs that are real, accessible, and contain the claimed information. | > 95% |
| Answer relevance | % of responses that address the actual coverage question asked. Scored by RAGAS AnswerRelevancy. | > 85% |
| Context relevance | % of retrieved chunks that are relevant to the question asked. Scored by RAGAS LLMContextPrecisionWithoutReference. | > 80% |
| Empty retrieval rate | % of queries where no relevant policy is returned. | < 15% |
| False coverage rate | % of responses that state "covered" for a service that is not covered per the cited policy. Manually audited on 10% sample. | **0% — critical failure mode** |

### 4.2 Operational metrics

| Metric | Definition | Target |
|---|---|---|
| Query response time p95 | End-to-end latency from submission to first token. | < 8 seconds |
| Cost per query | Embedding + generation + judge. | < $0.06 |
| Human review escalation rate | % of queries routed to human reviewer. | < 10% |
| Reviewer correction rate | % of human-reviewed responses overridden. | Track only (corrections = training data) |

---

## 5. Evaluation Framework

### 5.1 Golden dataset

| Component | Specification |
|---|---|
| Dataset size | 198 Q&A pairs generated via `src/evaluation/generate_golden.py`. One question + reference answer per sampled NCD or LCD document. Stored in `data/golden_dataset.json`. Target 500 by Month 3. |
| Fields | `question`, `reference_answer`, `policy_number`, `title`, `source`, `document_type` ("NCD" or "LCD"), `requires_jurisdiction` (bool — true for all LCDs). |
| Ground truth source | Synthetically generated — llama-3.1-8b-instant (Groq) reads 3,000 chars of each policy and generates one answerable question + grounded reference answer. |
| LCD entries | 119/198 entries are LCDs (`requires_jurisdiction=true`). These are **excluded from eval runs** in Phase 1 until full jurisdiction handling is implemented. Eval log reports coverage gap %. |
| NCD entries | 79/198. These are the active eval set for Phase 1. |
| Answer cache | `logs/rag_answers_cache.json` — persists RAG answers across runs. Questions already answered are never re-fetched. Prevents wasting free-tier API quota on repeat calls. |
| Special test cases required | NCD/LCD conflict for same procedure; CED requirements; geographic variation across MACs; vocabulary gap (clinical term ≠ CMS policy term). |

### 5.2 Automated evaluation pipeline

`src/evaluation/judge.py` runs RAGAS evaluation against the NCD subset of the golden dataset:

1. Load golden dataset → filter out `requires_jurisdiction=true` entries → log coverage gap
2. Load answer cache → skip already-answered questions
3. Call `rag_answer()` for each uncached NCD question (7s inter-call delay; indefinite retry on rate limits using API `retryDelay`)
4. Run RAGAS per-sample: Faithfulness + AnswerRelevancy + ContextRelevance (7s between samples)
5. Append scores + timestamp to `logs/eval_results.jsonl`

---

## 6. Data Strategy & Flywheel

### 6.1 Phase 1 data sources (all public)

- **NCDs:** ~400 documents via `api.coverage.cms.gov/v1/reports/national-coverage-ncd`
- **LCDs:** ~2,000+ documents via `api.coverage.cms.gov/v1/reports/local-coverage-final-lcds`
- **Ingestion targets:**
  - Week 1: All NCDs fully indexed with metadata
  - Week 2: All LCDs indexed with MAC region metadata
  - Week 3: Medicare Benefit Policy Manual chapters + CMS MLN articles for vocabulary bridging
- **Refresh cadence:** Weekly automated diff-check against CMS database. Re-index updated documents within 48 hours.

### 6.2 Data flywheel design

| Signal | What it means | How it feeds back |
|---|---|---|
| Reviewer clicks through to cms.gov link | Answer was useful enough to verify | Log with query + response |
| Reviewer copies answer text | Answer is being used | Log clipboard event |
| Reviewer triggers human review escalation | Output not trusted | Log full session as correction candidate |
| Reviewer types follow-up immediately | Answer was incomplete | Follow-up + original context = new golden pair |
| Session abandoned < 15s after response | Answer wrong or irrelevant | Log as negative training candidate |

---

## 7. Human-in-the-Loop (HITL) Design

### 7.1 HITL checkpoints

- **Faithfulness gate:** If RAGAS faithfulness score < threshold, route to human reviewer instead of delivering response.
- **Empty retrieval:** If no relevant policy found, return structured "no policy found" message and route to medical policy team.
- **Reviewer corrections:** Corrections logged and tagged for golden dataset expansion and eventual fine-tuning (Phase 4).

---

## 8. Failure Modes & Mitigations

| Failure mode | How it manifests | Mitigation |
|---|---|---|
| Wrong NCD/LCD retrieved | Agent retrieves policy for similar but different procedure. | Display full NCD/LCD title and ID before synthesis. Reviewer must verify source. |
| Outdated policy cited | NCD updated but old version still in vector DB. | Weekly diff-check. Surface version date prominently. |
| Conditional coverage missed | "Covered if A and B" synthesised as simply "covered". | Structured output requires explicit conditional criteria field. Judge evaluates condition preservation. |
| Vocabulary gap mismatch | Reviewer asks "Ozempic for obesity"; policy refers to "semaglutide, HCPCS J0223". | Index CMS MLN articles as vocabulary bridge. Synonym expansion before embedding. |
| Geographic variation missed | LCD retrieved applies to Jurisdiction 15 only; reviewer in Jurisdiction 5 applies it incorrectly. | **Prompt instruction (v1.2):** LCD responses include jurisdiction warning. Full UI enforcement deferred to Phase 2. |
| PHI in query | Reviewer pastes full prior auth request with member name/DOB. | Input validation screens for PHI before any data sent to LLM or stored. |
| Prompt injection via policy document | Malicious/corrupt NCD/LCD text overrides coverage determination. | Document sanitisation at ingestion. Output validation before delivery. |

---

## 9. UX Considerations

### 9.1 Design principles

- **Verification before trust:** Show which NCD/LCD was retrieved before showing the synthesised answer.
- **Explicit uncertainty:** "No matching Medicare coverage policy found" is a correct, useful output.
- **Framing discipline:** Every response must include: *"This output is for informational support only. Coverage determinations require clinical judgment and must be reviewed by a qualified professional before any action is taken."*
- **Geographic salience:** MAC region must be visually prominent for LCD results. Reviewers habitually overlook geographic scope.

### 9.2 Latency handling

| Response time | UX approach |
|---|---|
| < 4s | Direct delivery. |
| 4–10s | Show retrieved NCD/LCD titles immediately after retrieval (before synthesis). |
| > 10s | Progress indicator: Retrieving → Synthesising → Verifying. Stream tokens once generation begins. |
| Faithfulness check fails | Surface raw sources with links. Never hide the failure. |

---

## 10. FinOps & Cost Model

> NCD and LCD documents are long (5,000–10,000 words). Input token costs are higher than a typical RAG pipeline. Model selection and context management are the primary cost levers.

| Step | Model | Estimated cost per query |
|---|---|---|
| Query embedding | `BAAI/bge-small-en-v1.5` — local CPU | $0.00 |
| Vector search | ChromaDB — local | $0.00 |
| Answer generation — input | `gemini-2.5-flash` (~$0.30/1M input tokens, ~8K tokens context) | ~$0.0024 |
| Answer generation — output | `gemini-2.5-flash` (~$2.50/1M output tokens, ~400 tokens) | ~$0.001 |
| RAGAS judge — input | `gemini-2.5-flash-lite` (~$0.10/1M input tokens, ~9K tokens) | ~$0.0009 |
| RAGAS judge — output | `gemini-2.5-flash-lite` (~$0.40/1M output tokens, ~200 tokens) | ~$0.00008 |
| **Total per query** | | **~$0.004** |

---

## 11. Safety & Compliance

### 11.1 Safety requirements

- No PHI may be sent to any external API. Input validation blocks PHI before LLM call.
- Every response must cite the source NCD/LCD by policy number and title.
- System prompt explicitly prohibits generating coverage opinions from model memory.
- Framing disclaimer required on every response.

### 11.2 Regulatory requirements

- CMS-0057-F (2024) PA transparency mandate: outputs must be documentable and auditable.
- All data sources are public CMS data — no HIPAA data in the pipeline.
- Interaction log (`logs/interactions.jsonl`) provides audit trail.

---

## 12. Phase 1 Roadmap

| Week | Target |
|---|---|
| 1 | NCD ingestion + indexing complete. Basic RAG chain answering NCD questions. |
| 2 | LCD ingestion + indexing. Streamlit UI live. Interaction logging. |
| 3 | RAGAS evaluation pipeline. Golden dataset (200 pairs). Faithfulness > 90% on NCD subset. |
| 4 | Rate limit hardening (Gemini free tier). Persistent answer cache. LCD prompt instruction. Eval filtered to NCD-only with coverage gap logging. |
| Month 2 | Expert review of 20 golden pairs/month. LCD jurisdiction UI enforcement (see Phase 2). Vocab bridge (CMS MLN articles). |
| Month 3 | Golden dataset → 500 pairs. Citation link-checking automation. False coverage rate audit. |

---

---

# PHASE 2 — Evidence vs Coverage Gap Analyzer

---

## 13. Phase 2 Overview

Phase 2 solves the strategic problem: *"Where does what CMS covers diverge from what the evidence supports?"*

Phase 1 answers: *"What does CMS cover for this?"*
Phase 2 answers: *"Where is CMS coverage out of step with published clinical evidence — and what is the delta?"*

This is a materially harder problem requiring agentic multi-source synthesis, not just retrieval. Phase 2 is introduced after Phase 1 achieves stable quality metrics (Faithfulness > 90%, False Coverage Rate = 0%).

---

## 14. Phase 2 Problem Statement

Plans that rely purely on CMS policy miss coverage opportunities for members when:
- Strong RCT or meta-analysis evidence exists for a service CMS has not yet covered
- A CMS NCD/LCD was written before a pivotal trial published
- An LCD restricts coverage that the NCD allows, and the restriction is not evidence-based

Current state: medical policy teams manually review PubMed against CMS policy at $200K–500K per engagement via external consultants. This is periodic, not continuous.

Phase 2 automates systematic, continuous comparison of CMS policy against indexed clinical evidence.

---

## 15. Phase 2 Architecture

### 15.1 Multi-source agentic pipeline

Phase 2 requires an agent loop because it must independently search two sources and synthesise a comparison:

1. **Source A:** RAG retrieval from CMS NCD/LCD corpus (Phase 1 pipeline, unchanged)
2. **Source B:** PubMed/MEDLINE search for clinical evidence on the same service/indication
3. **Synthesis agent:** Compare coverage criteria from Source A against evidence strength from Source B. Identify gaps, conflicts, and alignment.
4. **Gap report:** Structured output — coverage position, evidence grade, gap type (CMS lags evidence / evidence supports CMS / evidence conflicts with CMS), recommended action.

### 15.2 LCD jurisdiction — full implementation

Phase 2 delivers full LCD jurisdiction handling:

- **MAC region metadata** surfaced in UI for all LCD responses — visually prominent, not dismissible
- **Reviewer jurisdiction profile** — reviewer's MAC region stored in session; LCDs from other jurisdictions flagged automatically
- **LCD-specific eval coverage** — `requires_jurisdiction=true` entries re-admitted to eval dataset with jurisdiction-aware scoring
- **Jurisdiction filtering in retrieval** — option to filter retrieved LCDs by reviewer's MAC region

> This replaces the v1.2 prompt instruction workaround. The prompt instruction remains in place as a safety backstop even after full implementation.

---

## 16. Phase 2 Data Sources

| Source | Coverage | Access |
|---|---|---|
| CMS NCDs/LCDs | ~400 NCDs, ~2,000+ LCDs | CMS Coverage API (Phase 1, already indexed) |
| PubMed/MEDLINE | 35M+ abstracts | NCBI E-utilities API (free, rate-limited) |
| ClinicalTrials.gov | ~500K trials | ClinicalTrials API v2 (free) |
| CMS LCD comment files | Stakeholder evidence submissions | CMS Coverage API — supplementary documents |

---

## 17. Phase 2 Metrics

| Metric | Definition | Threshold |
|---|---|---|
| Gap identification accuracy | % of identified evidence-coverage gaps confirmed by medical policy analyst review | > 80% |
| False positive gap rate | % of flagged gaps that are not real gaps on expert review | < 20% |
| Evidence grade accuracy | % of evidence grades (RCT / meta-analysis / observational / case series) correctly classified | > 90% |
| Coverage gap report actionability | % of gap reports that result in a policy review action | Track only (Month 6+) |

---

## 18. Phase 2 Roadmap

| Milestone | Target |
|---|---|
| PubMed ingestion + indexing | Month 4 |
| Multi-source retrieval (CMS + PubMed) | Month 4 |
| Gap synthesis agent (first version) | Month 5 |
| LCD jurisdiction full implementation | Month 5 |
| Gap report structured output + UI | Month 5 |
| Expert validation of gap reports (20/month) | Month 6+ |
| Fine-tuning on gap assessments | Month 7+ |

---

*Medicare Coverage Intelligence Platform · PRD v1.2 · All data sources public · Last updated 2026-06-07*
