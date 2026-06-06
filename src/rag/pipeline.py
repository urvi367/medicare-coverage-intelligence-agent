"""RAG pipeline: retrieve CMS coverage docs and generate answers with Groq."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
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


def answer(
    question: str,
    model: str = "llama-3.3-70b-versatile",
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve relevant policy chunks and return a cited answer via Groq.

    Returns a dict with keys:
        answer  — the generated response string
        sources — list of source Documents used
    """
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    sources: list[Document] = retriever.invoke(question)
    context = _format_docs(sources)
    prompt_value = _PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(prompt_value)

    return {"answer": response.content, "sources": sources}


def build_chain(model: str = "llama-3.3-70b-versatile", k: int = 5):
    """Return a streaming-compatible LangChain LCEL chain (answer text only)."""
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
