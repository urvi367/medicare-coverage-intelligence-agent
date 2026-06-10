"""RAG pipeline: retrieve CMS coverage docs and generate answers with Gemini."""

import logging
import random
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
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
}

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def _rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """Score (query, doc) pairs with a cross-encoder and return the top_n docs."""
    if not docs:
        return docs
    pairs = [(query, d.page_content) for d in docs]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

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
    db = load_index()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k or PIPELINE_CONFIG["k"], "score_threshold": PIPELINE_CONFIG["threshold"]},
    )
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    sources: list[Document] = _rerank(
        question, retriever.invoke(question), top_n=PIPELINE_CONFIG["reranker_top_n"]
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


_GAP_SYSTEM = (
    "You are a Medicare coverage policy expert comparing CMS policy to published clinical evidence.\n\n"
    "You have two sets of documents:\n"
    "  1. CMS POLICY — NCDs/LCDs stating Medicare's official coverage position\n"
    "  2. PUBMED EVIDENCE — peer-reviewed abstracts on clinical outcomes\n\n"
    "Structure your response exactly as follows:\n\n"
    "CMS Coverage Position: [Covered | Not Covered | Covered with Conditions | Not Addressed]\n"
    "  - [NCD/LCD number]: [criteria exactly as written]\n\n"
    "Clinical Evidence:\n"
    "  - [PMID year, journal]: [key finding and study type — 1 sentence]\n"
    "  (one bullet per abstract; write 'No relevant abstracts retrieved' if none)\n\n"
    "Evidence Grade: [Strong — RCT or meta-analysis | Moderate — cohort or observational | Weak / Insufficient]\n\n"
    "Alignment: [Aligned | Partially Aligned | Conflicting | "
    "Coverage Gap — evidence supports but CMS does not cover | "
    "Inverse Gap — CMS covers but clinical evidence is weak]\n\n"
    "Gap Summary: [2–3 sentences: where CMS policy and evidence agree or diverge, "
    "and the practical implication for coverage decisions.]\n\n"
    "CMS Policy Documents:\n{policy_context}\n\n"
    "PubMed Abstracts:\n{pubmed_context}"
)


def _format_pubmed_docs(docs: list[Document]) -> str:
    """Format PubMed abstracts into a numbered context block."""
    if not docs:
        return "No PubMed abstracts retrieved."
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = f"[{i}] PMID {m.get('pmid', '?')} ({m.get('year', '?')}) — {m.get('journal', '?')}"
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
    threshold = PIPELINE_CONFIG["threshold"]

    policy_db = load_index()
    policy_docs: list[Document] = _rerank(
        question,
        policy_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k_, "score_threshold": threshold},
        ).invoke(question),
        top_n=PIPELINE_CONFIG["reranker_top_n"],
    )

    try:
        pubmed_db = load_pubmed_index()
        pubmed_docs: list[Document] = _rerank(
            question,
            pubmed_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k_, "score_threshold": threshold},
            ).invoke(question),
            top_n=PIPELINE_CONFIG["reranker_top_n"],
        )
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


def build_chain(model: str = "gemini-2.5-flash", k: int | None = None):
    """Return a streaming-compatible LangChain LCEL chain (answer text only)."""
    db = load_index()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k or PIPELINE_CONFIG["k"], "score_threshold": PIPELINE_CONFIG["threshold"]},
    )
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def _build_prompt(inputs: dict):
        docs = _rerank(inputs["question"], inputs["docs"], top_n=PIPELINE_CONFIG["reranker_top_n"])
        prompt = ChatPromptTemplate.from_messages(
            [("system", _build_system(docs)), ("human", "{question}")]
        )
        return prompt.format_messages(context=_format_docs(docs), question=inputs["question"])

    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(_build_prompt)
        | llm
        | StrOutputParser()
    )
