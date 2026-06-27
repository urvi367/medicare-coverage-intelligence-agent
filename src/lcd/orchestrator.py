"""Agentic orchestration for dynamic LCD lookup (Phase 3, step 3).

A Gemini function-calling loop drives DETERMINISTIC tools — it decides *when* to
call which, and handles the branches/fallbacks; the tools' results are not
LLM-judged. Flow the model is instructed to follow:

    resolve_governing_ncd(service)
      governs  → answer from the national NCD (no LCD lookup)
      defers / silent → coverage is local:
          need the beneficiary's state; ASK for it if unknown, else
          resolve_mac_for_state(state)
            unsupported → report (out-of-scope MAC)
            mac → lookup_lcds(mac, service)
                  found  → answer from the live LCD(s) (jurisdiction-scoped)
                  no_lcd → contractor discretion / individual consideration
                  failed → degrade gracefully, suggest verifying

Entry point: `run_coverage_query(query, state=None)` -> {answer, trace, ...}.
When the model needs a state it doesn't have, it asks; the caller re-invokes
with `state=` set from the user's reply.
"""
from __future__ import annotations

import json
import logging
import math

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.lcd.jurisdiction import SUPPORTED_MACS, extract_state, ncd_disposition, resolve_mac
from src.lcd.lookup import LcdLookupError, lcd_lookup
from src.rag.pipeline import (
    PIPELINE_CONFIG,
    _full_ncd_docs,
    _hybrid_retrieve_ncd,
    _rerank_scored,
    _select_primary_ncds,
)

logger = logging.getLogger(__name__)

SILENT_GATE = 0.60   # top-NCD rerank sigmoid below this ⇒ no NCD governs ⇒ go local
_MAX_STEPS = 6


# ── deterministic tools ──────────────────────────────────────────────────────
@tool
def resolve_governing_ncd(service: str) -> dict:
    """Determine whether a National Coverage Determination governs a clinical service.

    Always call this FIRST. `service` is the clinical service/procedure from the
    question (e.g. "panniculectomy", "CPAP for sleep apnea"). Returns disposition:
    "governs" (a national NCD covers it — answer from the NCD), "defers" (an NCD
    exists but hands coverage to local MACs), or "silent" (no NCD — coverage is local).
    """
    reranked = _rerank_scored(
        service, _hybrid_retrieve_ncd(service, PIPELINE_CONFIG["k"]),
        top_n=PIPELINE_CONFIG["reranker_top_n"],
    )
    if not reranked:
        return {"disposition": "silent"}
    top_sig = 1.0 / (1.0 + math.exp(-reranked[0][0]))
    if top_sig < SILENT_GATE:
        return {"disposition": "silent", "top_relevance": round(top_sig, 3)}
    primary = _select_primary_ncds(reranked)  # deterministic score-weighted vote
    if not primary:
        return {"disposition": "silent"}
    docs = _full_ncd_docs(primary[0])
    text = "\n".join(d.page_content for d in docs)
    disp = ncd_disposition(text)
    out = {
        "disposition": disp,
        "ncd_number": primary[0],
        "ncd_title": docs[0].metadata.get("title", "") if docs else "",
    }
    if disp == "governs":
        out["ncd_text"] = text[:3500]
    return out


@tool
def resolve_mac_for_state(state: str) -> dict:
    """Map a US state (name or 2-letter code) to its MAC contractor.

    Call this only when coverage is local (disposition defers/silent) AND you know
    the beneficiary's state. Returns {"mac": name} when in scope, or a status of
    "unsupported_jurisdiction" (a real state served by an out-of-scope MAC) /
    "unknown_state" (couldn't parse a state).
    """
    code = extract_state(state) or (state.strip().upper() if state else "")
    mac = resolve_mac(code)
    if mac:
        return {"mac": mac, "state": code}
    if code:
        return {"status": "unsupported_jurisdiction", "state": code,
                "supported_macs": list(SUPPORTED_MACS)}
    return {"status": "unknown_state"}


@tool
def lookup_lcds(mac: str, service: str) -> dict:
    """Fetch the live LCD(s) for a MAC + clinical service (dynamic, query-time).

    Call only after resolve_mac_for_state returns a mac. `service` should be a
    DESCRIPTIVE clinical phrase including common synonyms (e.g. "panniculectomy,
    excess skin / abdominal reconstructive surgery"), not a single bare word — LCD
    titles are broad, so a richer phrase matches far better. Returns status "found"
    with the LCD(s) and their text, "no_lcd" (none in this jurisdiction → contractor
    discretion), or "lookup_failed" (the LCD service was unreachable).
    """
    try:
        matches = lcd_lookup(mac, service)
    except LcdLookupError as e:
        return {"status": "lookup_failed", "error": str(e)}
    if not matches:
        return {"status": "no_lcd", "mac": mac}
    return {"status": "found", "mac": mac, "lcds": [
        {"lcd_id": m.lcd_id, "title": m.title, "relevance": m.score, "text": m.text[:3000]}
        for m in matches
    ]}


_TOOLS = {t.name: t for t in (resolve_governing_ncd, resolve_mac_for_state, lookup_lcds)}


def _as_text(content) -> str:
    """Gemini may return content as a string or a list of parts — flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return str(content)

_SYSTEM = (
    "You are a Medicare coverage assistant. Answer whether a service is covered, using "
    "ONLY the tools and the policy text they return — never your own knowledge of coverage.\n\n"
    "Procedure:\n"
    "1. Call resolve_governing_ncd with the clinical service from the question.\n"
    "2. If disposition is 'governs': answer from the returned NCD text (national coverage). "
    "Do NOT look up LCDs.\n"
    "3. If 'defers' or 'silent': coverage is set locally by the Medicare Administrative "
    "Contractor (MAC), which depends on the beneficiary's STATE.\n"
    "   - Known state for this query: {state}.\n"
    "   - If the state is 'unknown', ASK the user which US state the beneficiary is in, and "
    "STOP — do not call other tools.\n"
    "   - Otherwise call resolve_mac_for_state(state).\n"
    "       • 'unsupported_jurisdiction' → tell the user this assistant currently covers only "
    "these MACs: {macs}, and their state isn't among them.\n"
    "       • a mac → call lookup_lcds(mac, service).\n"
    "           – 'found' → answer from the LCD text; state the LCD id and that it applies to "
    "that jurisdiction only.\n"
    "           – 'no_lcd' → there is no LCD for this service in that jurisdiction; coverage is "
    "at contractor discretion / by individual consideration.\n"
    "           – 'lookup_failed' → the LCD service is temporarily unavailable; advise verifying "
    "directly on the MCD. Do not fabricate coverage.\n"
    "Always cite the NCD/LCD id you used. Be concise."
)


def run_coverage_query(query: str, state: str | None = None) -> dict:
    """Run the agentic NCD→LCD coverage loop for one query.

    Args:
        query: the user's coverage question.
        state: optional known state (e.g. from a prior turn's answer to "which state?").

    Returns {"answer": str, "trace": [tool calls], "needs_state": bool}.
    """
    known = state or extract_state(query) or "unknown"
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(
        list(_TOOLS.values())
    )
    messages = [
        SystemMessage(_SYSTEM.format(state=known, macs=", ".join(SUPPORTED_MACS))),
        HumanMessage(query),
    ]
    trace: list[dict] = []
    for _ in range(_MAX_STEPS):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            escalated = any(t["tool"] == "resolve_governing_ncd"
                            and t["result"] in ("defers", "silent") for t in trace)
            proceeded = any(t["tool"] in ("resolve_mac_for_state", "lookup_lcds") for t in trace)
            needs_state = known == "unknown" and escalated and not proceeded
            return {"answer": _as_text(ai.content), "trace": trace, "needs_state": needs_state}
        for tc in ai.tool_calls:
            result = _TOOLS[tc["name"]].invoke(tc["args"])
            trace.append({"tool": tc["name"], "args": tc["args"],
                          "result": result.get("status") or result.get("disposition")
                          or result.get("mac")})
            messages.append(ToolMessage(content=json.dumps(result)[:8000], tool_call_id=tc["id"]))
    return {"answer": "(unresolved — max tool steps reached)", "trace": trace, "needs_state": False}
