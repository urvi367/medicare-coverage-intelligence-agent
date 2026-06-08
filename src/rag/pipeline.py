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
from langchain_core.runnables import RunnablePassthrough

from src.rag.indexer import load_index

load_dotenv()
logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a Medicare coverage policy expert. Answer questions using ONLY the "
    "retrieved policy documents below. For every claim, cite the document title and "
    "policy number. If the documents do not contain enough information to answer "
    "confidently, say so explicitly.\n\n"
    "If any retrieved document is an LCD (Local Coverage Determination), explicitly "
    "state at the start of your answer: 'Note: This determination is based on an LCD "
    "which applies to [jurisdiction] only. Coverage may differ in other MAC regions.' "
    "If the jurisdiction is not known, say the jurisdiction is unknown and the user "
    "must verify.\n\n"
    "If both an NCD and LCD are retrieved for the same service, clearly distinguish "
    "them: state the NCD national coverage position first, then state how the LCD adds "
    "or modifies criteria for the specific jurisdiction.\n\n"
    "Retrieved documents:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", "{question}")]
)


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
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve relevant policy chunks and return a cited answer via Gemini.

    Returns a dict with keys:
        answer  — the generated response string
        sources — list of source Documents used
    """
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    sources: list[Document] = retriever.invoke(question)
    context = _format_docs(sources)
    prompt_value = _PROMPT.format_messages(context=context, question=question)

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


def build_chain(model: str = "gemini-2.5-flash", k: int = 5):
    """Return a streaming-compatible LangChain LCEL chain (answer text only)."""
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
