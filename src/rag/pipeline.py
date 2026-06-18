"""RAG pipeline: retrieve CMS coverage docs and generate answers with Gemini."""

import logging
import math
import random
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sentence_transformers import CrossEncoder

from src.rag.indexer import load_index
from src.rag.pubmed_indexer import load_pubmed_index

PIPELINE_CONFIG = {
    "k": 10,
    "threshold": 0.65,
    "reranker": "BAAI/bge-reranker-base",
    "reranker_top_n": 5,
    "search_mode": "hybrid",
    "pubmed_k": 12,  # abstracts pulled per NCD topic for gap analysis (matches the
                     # labeler's evidence budget so the pipeline sees the same evidence)
}

_reranker: CrossEncoder | None = None
_bm25: BM25Retriever | None = None
_db = None  # cached Chroma instance — avoid reopening on every query
_pubmed_db = None  # cached PubMed Chroma instance


def _get_db():
    global _db
    if _db is None:
        _db = load_index()
    return _db


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def _get_pubmed_db():
    global _pubmed_db
    if _pubmed_db is None:
        _pubmed_db = load_pubmed_index()
    return _pubmed_db


def _get_bm25(k: int) -> BM25Retriever:
    """Build (once) and return a BM25 retriever over the full CMS index."""
    global _bm25
    if _bm25 is None:
        result = _get_db().get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        _bm25 = BM25Retriever.from_documents(docs)
        logger.info("BM25 index built over %d CMS chunks", len(docs))
    _bm25.k = k
    return _bm25


def _hybrid_retrieve(query: str, k: int) -> list[Document]:
    """Fuse BM25 (sparse) + dense vector results via Reciprocal Rank Fusion."""
    db = _get_db()
    dense_docs = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": PIPELINE_CONFIG["threshold"]},
    ).invoke(query)
    bm25_docs = _get_bm25(k).invoke(query)

    # RRF constant k=60 is standard; higher = smoother rank blending
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k_] for k_ in ranked[:k]]


def _hybrid_retrieve_ncd(query: str, k: int) -> list[Document]:
    """Hybrid retrieve restricted to NCD chunks only (for gap analysis CMS side)."""
    db = _get_db()
    dense_docs = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": PIPELINE_CONFIG["threshold"],
            "filter": {"source": "NCD"},
        },
    ).invoke(query)
    # BM25 has no native filter — post-filter after scoring
    bm25_docs = [
        d for d in _get_bm25(k).invoke(query)
        if d.metadata.get("source") == "NCD"
    ]

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k_] for k_ in ranked[:k]]


def _pubmed_for_ncds(query: str, ncd_numbers: set[str], k: int) -> list[Document]:
    """Retrieve PubMed abstracts restricted to specific NCD topics, ranked by query.

    Guarantees topical alignment: every abstract was originally fetched for one of
    the NCDs surfaced on the policy side (matched via source_ncd_number == the NCD's
    policy_number), so the evidence set and the coverage position describe the same
    intervention. No score threshold — the NCD filter already enforces topicality;
    we only rank the small per-NCD abstract pool by query relevance.
    """
    if not ncd_numbers:
        return []
    db = _get_pubmed_db()
    nums = list(ncd_numbers)
    flt = {"source_ncd_number": nums[0]} if len(nums) == 1 else {"source_ncd_number": {"$in": nums}}
    return db.similarity_search(query, k=k, filter=flt)


def _rerank_scored(query: str, docs: list[Document], top_n: int) -> list[tuple[float, Document]]:
    """Score (query, doc) pairs with a cross-encoder; return top_n as (score, doc)."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(float(s), d) for s, d in ranked[:top_n]]


def _rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """Score (query, doc) pairs with a cross-encoder and return the top_n docs."""
    return [doc for _, doc in _rerank_scored(query, docs, top_n)]


def _full_ncd_docs(ncd_number: str) -> list[Document]:
    """Return ALL indexed chunks for one NCD.

    A coverage determination is a single document; ranking its internal chunks can
    drop the eligibility-criteria section (the part that distinguishes Partial gaps
    from Aligned), so for gap analysis we feed the whole policy rather than only the
    chunks most similar to the question.
    """
    got = _get_db().get(
        where={"policy_number": ncd_number}, include=["documents", "metadatas"]
    )
    return [
        Document(page_content=t, metadata=m)
        for t, m in zip(got["documents"], got["metadatas"])
    ]


def _select_primary_ncds(
    scored: list[tuple[float, Document]], second_frac: float = 0.7
) -> list[str]:
    """Collapse reranked NCD chunks to the primary NCD by score-weighted vote.

    Sum each NCD's reranker relevance (sigmoid of the cross-encoder logit, so the
    threshold is meaningful and stray negative-scored chunks don't dominate) across
    the reranked chunks, and pick the top NCD. A SECOND NCD is included only when its
    aggregate score is within `second_frac` of the top — the rare case of genuine
    co-governance (a general NCD + a sub-NCD). Otherwise gap analysis reasons over a
    single policy, which both restores the criteria text and drops weakly-relevant
    stray NCDs that would otherwise contaminate the verdict.
    """
    agg: dict[str, float] = {}
    for score, d in scored:
        n = d.metadata.get("policy_number")
        if not n:
            continue
        agg[n] = agg.get(n, 0.0) + 1.0 / (1.0 + math.exp(-score))
    if not agg:
        return []
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    top_n, top_s = ranked[0]
    chosen = [top_n]
    # Add at most ONE more NCD — the runner-up, and only if it's within second_frac of
    # the top. Capping at two is deliberate: when many NCDs score similarly the result
    # is an ambiguous query, not genuine co-governance, and pulling 3-5 full policies in
    # would reintroduce exactly the cross-policy contamination this design removes.
    if len(ranked) > 1:
        runner_n, runner_s = ranked[1]
        if top_s > 0 and runner_s >= second_frac * top_s:
            chosen.append(runner_n)
    return chosen

load_dotenv()
logger = logging.getLogger(__name__)

_BASE_SYSTEM = (
    "You are a Medicare coverage policy expert. Answer questions using ONLY the "
    "retrieved policy documents below. For every claim, cite the document title and "
    "policy number. If the documents do not contain enough information to answer "
    "confidently, say so explicitly."
)

_LCD_ADDENDUM = (
    "\n\nOne or more retrieved documents are LCDs (Local Coverage Determinations). "
    "State at the start of your answer: 'Note: This determination is based on an LCD "
    "which applies to [jurisdiction] only. Coverage may differ in other MAC regions.' "
    "If jurisdiction is unknown, say so and instruct the user to verify. "
    "If both an NCD and LCD are retrieved for the same service, state the NCD national "
    "coverage position first, then how the LCD modifies criteria for that jurisdiction."
)


def _build_system(docs: list[Document]) -> str:
    """Return a system prompt with the LCD note only when the top-ranked doc is an LCD.

    Stray LCD chunks ranked 2nd-5th (cosine overlap on a related topic) should not
    trigger a jurisdiction warning on what is otherwise an NCD answer.
    """
    has_lcd = bool(docs) and docs[0].metadata.get("source", "") == "LCD"
    base = _BASE_SYSTEM
    if has_lcd:
        base += _LCD_ADDENDUM
    return base + "\n\nRetrieved documents:\n{context}"


def _format_docs(docs: list[Document]) -> str:
    """Format a list of Documents into a numbered context block."""
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = f"[{i}] {m.get('source', '')} {m.get('policy_number', '')} — {m.get('title', '')}"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_retry_delay(exc: BaseException) -> float | None:
    """Extract the API-suggested retryDelay (seconds) from a Gemini error, if present."""
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
    return float(m.group(1)) if m else None


def _retryable(exc: BaseException) -> bool:
    """Return True for Gemini errors the API expects the client to retry.

    The presence of retryDelay in the error body is the authoritative signal —
    use it regardless of whether the quota metric name mentions PerDay.
    """
    s = str(exc)
    if "retryDelay" in s:
        return True
    if isinstance(exc, ChatGoogleGenerativeAIError):
        return "429" in s or "RESOURCE_EXHAUSTED" in s or "503" in s or "SERVICE_UNAVAILABLE" in s
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and val in (429, 503):
            return True
    sl = s.lower()
    return any(t in sl for t in ("429", "rate limit", "too many requests",
                                  "resource exhausted", "service unavailable"))


def answer(
    question: str,
    model: str = "gemini-2.5-flash",
    k: int | None = None,
) -> dict[str, Any]:
    """Retrieve relevant policy chunks and return a cited answer via Gemini.

    Returns a dict with keys:
        answer  — the generated response string
        sources — list of source Documents used
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    sources: list[Document] = _rerank(
        question,
        _hybrid_retrieve(question, k or PIPELINE_CONFIG["k"]),
        top_n=PIPELINE_CONFIG["reranker_top_n"],
    )
    context = _format_docs(sources)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _build_system(sources)), ("human", "{question}")]
    )
    prompt_value = prompt.format_messages(context=context, question=question)

    attempt = 0
    while True:
        try:
            response = llm.invoke(prompt_value)
            return {"answer": response.content, "sources": sources}
        except Exception as exc:
            if not _retryable(exc):
                raise
            suggested = _parse_retry_delay(exc)
            # Cap fallback backoff at 120s; use API's retryDelay when available.
            wait = (suggested + random.uniform(1, 3)) if suggested else min(2 ** min(attempt, 5) * 5 + random.uniform(0, 2), 120)
            logger.warning(
                "Gemini rate limited — waiting %.0fs before retry #%d",
                wait, attempt + 1,
            )
            time.sleep(wait)
            attempt += 1


def build_chain(model: str = "gemini-2.5-flash", k: int | None = None):
    """Return a streaming-compatible LangChain LCEL chain (answer text only)."""
    k_ = k or PIPELINE_CONFIG["k"]
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def _retrieve_and_rerank(question: str) -> list[Document]:
        return _rerank(question, _hybrid_retrieve(question, k_), top_n=PIPELINE_CONFIG["reranker_top_n"])

    def _build_prompt(inputs: dict):
        docs = inputs["docs"]
        prompt = ChatPromptTemplate.from_messages(
            [("system", _build_system(docs)), ("human", "{question}")]
        )
        return prompt.format_messages(context=_format_docs(docs), question=inputs["question"])

    return (
        {"docs": RunnableLambda(_retrieve_and_rerank), "question": RunnablePassthrough()}
        | RunnableLambda(_build_prompt)
        | llm
        | StrOutputParser()
    )


_GAP_SYSTEM = (
    "You are a Medicare coverage policy expert comparing CMS policy to published clinical evidence.\n\n"
    "You have two sets of documents:\n"
    "  1. CMS POLICY — NCDs (National Coverage Determinations) stating Medicare's official coverage position\n"
    "  2. PUBMED EVIDENCE — peer-reviewed abstracts on clinical outcomes\n\n"
    "ALWAYS reply in the exact structured format below — every section including an "
    "'Alignment:' line — even if the CMS policy text is a stub/redirect with no criteria, "
    "or the question cannot be analyzed as a coverage-vs-evidence comparison — including NCDs that "
    "set an administrative/payment/documentation condition rather than testing a clinical "
    "intervention's efficacy (e.g. when anesthesia is documented as medically necessary, or "
    "evaluation/management & consultation services). In those cases "
    "output 'Alignment: Insufficient Evidence — no clinical evidence retrieved' with a "
    "one-line Gap Summary explaining why. NEVER answer in free-form prose without the "
    "Alignment line.\n\n"
    "Structure your response exactly as follows:\n\n"
    "CMS Coverage Position: [Covered | Not Covered | Covered with Conditions | Not Addressed]\n"
    "  - [NCD number]: [criteria exactly as written]\n\n"
    "Clinical Evidence:\n"
    "  - PMID <number> (<year>, <journal>): [key finding and study type — 1 sentence]\n"
    "  ALWAYS begin each bullet with the literal token 'PMID' followed by the numeric\n"
    "  PubMed ID exactly as it appears in the abstracts (e.g. 'PMID 12811203').\n"
    "  (one bullet per abstract; write 'No relevant abstracts retrieved' if none)\n\n"
    "Evidence Grade: [Strong — T1/T2 (meta-analysis, systematic review, RCT) | Moderate — T3/T4 "
    "(clinical trial, cohort, observational) | Weak / Insufficient — T5/background (case report, "
    "review, guideline, unspecified)]\n"
    "  Use the evidence tier shown in brackets in each abstract's header (derived from the NLM\n"
    "  PublicationType). 'background only' (review/guideline) and 'unspecified' are NOT primary\n"
    "  evidence — grade them Weak unless the abstract text clearly describes a stronger design.\n\n"
    "Alignment: [Aligned — CMS and evidence agree | "
    "Partial Coverage Gap — CMS covers it but the evidence supports a substantive, "
    "clinically-distinct broader use it excludes (not a marginal restatement) | "
    "Coverage Gap — evidence supports but CMS does not cover or denies it | "
    "Overcoverage — CMS covers but the evidence is weak, absent, or negative | "
    "Insufficient Evidence — no relevant clinical evidence retrieved]\n"
    "  To choose the Alignment label, follow IN ORDER:\n"
    "  (a) If fewer than two abstracts genuinely study THIS intervention for THIS condition "
    "(after discarding name-collision abstracts) → Insufficient Evidence, stop.\n"
    "  (b) Establish CMS's position from the policy text — does CMS cover this service for ANY "
    "indication (fully, or only a narrow population/indication), or NOT cover it for any?\n"
    "  (c) Map by DIRECTION (which side is ahead), not by how CMS phrases its rationale:\n"
    "      • covers (any indication) AND evidence supports it → Aligned\n"
    "      • does NOT cover for any indication AND evidence does not support it → Aligned (agree)\n"
    "      • covers it but more narrowly than the evidence supports → Partial Coverage Gap\n"
    "        (POSITIVE TEST for Partial: the policy grants coverage with EXPLICIT eligibility "
    "criteria — a threshold/cutoff (e.g. LVEF ≤35%, an AHI/RDI value), an ENUMERATED list of "
    "covered indications/comorbidities/organs, or an FDA/compendia gate — AND the on-topic "
    "evidence supports efficacy in a clinically-distinct population that falls OUTSIDE those "
    "criteria (a comorbidity not listed, a broader severity/EF range, an additional indication "
    "or organ, a monitoring window that changes management). That excluded-but-eligible "
    "population is a coverage gap at the margin → Partial.\n"
    "        NOT Partial: a different graft source, delivery route, or device variant for the "
    "SAME covered indication is a technique variation → Aligned; a merely marginal restatement "
    "of the covered use → Aligned. Decide with this test: 'is there a concrete patient the "
    "policy's own written criteria would DENY, whom the evidence would treat?' — if yes → "
    "Partial; if the evidence only reinforces the already-covered population → Aligned.)\n"
    "      • does NOT cover it for ANY indication BUT evidence CONSISTENTLY supports it → Coverage Gap\n"
    "        (Coverage Gap needs reasonably consistent, clinically-meaningful evidence — RCT/meta-"
    "analysis quality. If CMS has a reasoned non-coverage and the positive evidence is Weak/Insufficient "
    "grade, internally mixed, or only marginally above sham/placebo, that agrees with non-coverage → "
    "Aligned, not Coverage Gap.)\n"
    "      • covers it BUT the on-topic evidence shows it does not work → Overcoverage\n"
    "  CRITICAL: if CMS covers the service for even ONE indication, the label can ONLY be "
    "Aligned, Partial Coverage Gap, or Overcoverage — NEVER a full Coverage Gap (reserved for "
    "services CMS covers for NO indication). 'Passively not covered' and 'actively denied citing "
    "weak evidence' are the SAME direction — both Coverage Gap when the evidence supports. "
    "Overcoverage requires the on-topic evidence to actually show it does not work; evidence "
    "merely absent → Insufficient Evidence, not Overcoverage. The ineffectiveness must apply to the "
    "COVERED population/indication as a whole — not just a narrow subgroup, one device/technique "
    "variant, an off-label use, or a dosing/threshold question (those are Partial or Aligned, not "
    "Overcoverage), and known side effects/surgical sequelae of an otherwise-indicated service are "
    "not 'does not work.'\n\n"
    "Gap Summary: [2–3 sentences: where CMS policy and evidence agree or diverge, "
    "and the practical implication for coverage decisions.]\n\n"
    "NAME-COLLISION CHECK: the abstracts were retrieved by the NCD's topic name, so some may "
    "be about a DIFFERENT intervention that merely shares the name or a keyword (e.g. a modern "
    "therapy with the same name as the obsolete one the policy describes, or a different device "
    "for the same organ). Read the CMS Policy text to learn what the intervention ACTUALLY is, "
    "and treat an abstract as evidence ONLY if it studies that SAME intervention for the SAME "
    "condition — not just a shared word. Ignore name-collision abstracts.\n\n"
    "IMPORTANT: If no PubMed abstracts are provided below ('No PubMed abstracts retrieved'), OR "
    "fewer than two of the provided abstracts genuinely study this intervention for this "
    "condition, you MUST set Evidence Grade to 'Insufficient' and Alignment to 'Insufficient "
    "Evidence — no clinical evidence retrieved'. Do not infer an alignment verdict without "
    "at least two on-topic abstracts.\n\n"
    "Cite ONLY PMIDs that appear in the PubMed Abstracts section below. Never invent a "
    "PMID, year, or finding not present in the provided abstracts.\n\n"
    "CMS Policy Documents:\n{policy_context}\n\n"
    "PubMed Abstracts:\n{pubmed_context}"
)


def _format_pubmed_docs(docs: list[Document]) -> str:
    """Format PubMed abstracts into a numbered context block."""
    if not docs:
        return "No PubMed abstracts retrieved."
    from src.ingestion.fetch_pubmed import evidence_tier

    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        stype = m.get("study_type") or "study type unspecified"
        tier = evidence_tier(m.get("study_type", ""))
        header = f"[{i}] PMID {m.get('pmid', '?')} ({m.get('year', '?')}, {stype} [{tier}]) — {m.get('journal', '?')}"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def gap_analysis(
    question: str,
    model: str = "gemini-2.5-flash",
    k: int | None = None,
) -> dict[str, Any]:
    """Retrieve CMS policy + PubMed evidence and return a structured gap report.

    Returns a dict with keys:
        gap_report     — structured gap analysis string
        policy_sources — CMS NCD/LCD Documents used
        pubmed_sources — PubMed abstract Documents used
    """
    k_ = k or PIPELINE_CONFIG["k"]

    # Use chunk retrieval only to IDENTIFY the governing NCD, then feed its full text.
    reranked = _rerank_scored(
        question,
        _hybrid_retrieve_ncd(question, k_),
        top_n=PIPELINE_CONFIG["reranker_top_n"],
    )
    primary_ncds = _select_primary_ncds(reranked)

    if primary_ncds:
        # Whole-NCD context: a coverage determination is one document, so supply the
        # complete policy (covered + non-covered indications + criteria) rather than
        # the handful of question-similar chunks, which can omit the criteria section.
        policy_docs: list[Document] = [d for n in primary_ncds for d in _full_ncd_docs(n)]
        ncd_numbers = set(primary_ncds)
    else:
        # Fallback (no policy_number on any chunk): keep the reranked chunks as-is.
        policy_docs = [d for _, d in reranked]
        ncd_numbers = {
            d.metadata.get("policy_number")
            for d in policy_docs
            if d.metadata.get("policy_number")
        }

    # Topical join: pull evidence only for the primary NCD(s), so the abstracts and
    # the coverage position are guaranteed to describe the same intervention.

    try:
        # Topical join only. If an NCD has no indexed abstracts we return nothing,
        # so the report yields "Insufficient Evidence" rather than comparing the
        # policy against unrelated abstracts pulled by an open (non-topical) search.
        pubmed_docs: list[Document] = _pubmed_for_ncds(question, ncd_numbers, k=PIPELINE_CONFIG["pubmed_k"])
        # Sort newest-first so Gemini weights recent evidence more heavily.
        pubmed_docs.sort(key=lambda d: d.metadata.get("year", "0"), reverse=True)
    except RuntimeError:
        logger.warning("PubMed index not found — run fetch_pubmed + pubmed_indexer first")
        pubmed_docs = []

    policy_context = _format_docs(policy_docs) if policy_docs else "No CMS policy documents retrieved."
    pubmed_context = _format_pubmed_docs(pubmed_docs)

    prompt_value = ChatPromptTemplate.from_messages(
        [("system", _GAP_SYSTEM), ("human", "{question}")]
    ).format_messages(
        policy_context=policy_context,
        pubmed_context=pubmed_context,
        question=question,
    )

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    attempt = 0
    while True:
        try:
            response = llm.invoke(prompt_value)
            return {
                "gap_report": response.content,
                "policy_sources": policy_docs,
                "pubmed_sources": pubmed_docs,
            }
        except Exception as exc:
            if not _retryable(exc):
                raise
            suggested = _parse_retry_delay(exc)
            wait = (suggested + random.uniform(1, 3)) if suggested else min(2 ** min(attempt, 5) * 5 + random.uniform(0, 2), 120)
            logger.warning("Gemini rate limited — waiting %.0fs before retry #%d", wait, attempt + 1)
            time.sleep(wait)
            attempt += 1
