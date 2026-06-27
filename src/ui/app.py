"""Streamlit chat UI for the Medicare Coverage Intelligence Agent."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag.pipeline import answer, gap_analysis
from src.lcd.jurisdiction import extract_state
from src.lcd.orchestrator import run_coverage_query

load_dotenv()
logging.basicConfig(level=logging.INFO)

LOG_DIR = Path(__file__).parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "interactions.jsonl"


# ── Query router ──────────────────────────────────────────────────────────────

# Signals that the user wants evidence vs policy comparison
_GAP_SIGNALS = re.compile(
    r"\b("
    r"evidence|clinical\s+evidence|evidence.?base[d]?|"
    r"research|study|studies|trial[s]?|clinical\s+trial[s]?|"
    r"literature|systematic\s+review|meta.?analys[ie]s|"
    r"rcts?|randomized|randomised|"
    r"pubmed|published|peer.?reviewed|journal|"
    r"finding[s]?|outcome[s]?|efficacy|effectiveness|"
    r"gap|coverage\s+gap|"
    r"compared?\s+to\s+(the\s+)?(evidence|research|literature|data)|"
    r"what\s+does\s+(the\s+)?(evidence|research|literature|data)\s+(say|show|suggest|support)|"
    r"is\s+there\s+(evidence|research|data)|"
    r"data\s+support[s]?|clinical\s+data|"
    r"does\s+(the\s+)?evidence|support[s]?\s+coverage"
    r")\b",
    re.IGNORECASE,
)

# Explicit policy signals (boost toward policy Q&A even if gap words appear)
_POLICY_SIGNALS = re.compile(
    r"\b("
    r"cover(ed|age|s)?|criteria|requirement[s]?|eligible|eligib[il]+ity|"
    r"medicare\s+(pay|reimburse|allow)|ncd|lcd|"
    r"medically\s+necessary|medical\s+necessity|"
    r"prior\s+auth|preauthori[sz]ation|"
    r"billing|claim|reimburs"
    r")\b",
    re.IGNORECASE,
)


# Jurisdictional signals → dynamic NCD→LCD coverage path (Phase 3). A LCD is
# jurisdictional, so an explicit state or local-coverage cue routes to the agentic
# coverage loop (which resolves NCD vs the state's MAC LCD, asking for state if needed).
_COVERAGE_SIGNALS = re.compile(
    r"\b(lcd|local\s+coverage|jurisdiction|mac\b|contractor|"
    r"in\s+my\s+state|my\s+state|which\s+state|by\s+state|in\s+my\s+(region|area))\b",
    re.IGNORECASE,
)


def _route(query: str) -> str:
    """Return 'gap', 'coverage', or 'policy' based on query signals."""
    gap_score = len(_GAP_SIGNALS.findall(query))
    policy_score = len(_POLICY_SIGNALS.findall(query))
    # Gap analysis only when evidence signals clearly dominate
    if gap_score > 0 and gap_score >= policy_score:
        return "gap"
    # Jurisdictional coverage question → dynamic NCD→LCD loop
    if extract_state(query) or _COVERAGE_SIGNALS.search(query):
        return "coverage"
    return "policy"


# ── Logging ───────────────────────────────────────────────────────────────────

_FEEDBACK_LABELS = {
    "positive": "✅ Yes, this answered my question",
    "partial":  "⚠️ Partially — missing some details",
    "negative": "❌ No, this seems incorrect",
}


_ALIGNMENT_RE = re.compile(r"Alignment:\s*\**(.+?)\**(?:\n|$)")


def _log_interaction(
    question: str,
    answer_text: str,
    sources: list,
    mode: str,
    pubmed_sources: list | None = None,
) -> None:
    policy_source_list = [
        {
            "title": s.metadata.get("title", ""),
            "policy_number": s.metadata.get("policy_number", ""),
            "source": s.metadata.get("source", ""),
            "excerpt": s.page_content,
        }
        for s in sources
    ]
    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "question": question,
        "answer": answer_text,
    }
    if mode == "gap":
        entry["policy_sources"] = policy_source_list
        m = _ALIGNMENT_RE.search(answer_text)
        if m:
            entry["alignment"] = m.group(1).strip()
        if pubmed_sources:
            entry["pubmed_sources"] = [
                {
                    "pmid": s.metadata.get("pmid", ""),
                    "title": s.metadata.get("title", ""),
                    "year": s.metadata.get("year", ""),
                    "journal": s.metadata.get("journal", ""),
                    "excerpt": s.page_content,
                }
                for s in pubmed_sources
            ]
    else:
        entry["sources"] = policy_source_list
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _log_feedback(question: str, rating: str, mode: str = "") -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "feedback",
        "mode": mode,
        "question": question,
        "rating": rating,
        "label": _FEEDBACK_LABELS[rating],
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _render_feedback(msg_index: int, question: str, mode: str = "") -> None:
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if msg_index in st.session_state.feedback:
        st.caption(f"Your feedback: {_FEEDBACK_LABELS[st.session_state.feedback[msg_index]]}")
        return
    cols = st.columns(3)
    for col, (key, label) in zip(cols, _FEEDBACK_LABELS.items()):
        if col.button(label, key=f"fb_{msg_index}_{key}"):
            st.session_state.feedback[msg_index] = key
            _log_feedback(question, key, mode=mode)
            st.rerun()


def _render_sources(sources: list[dict], label: str = "Sources") -> None:
    if not sources:
        return
    with st.expander(label):
        for s in sources:
            if s.get("pmid"):
                title = s.get("title") or f"PMID {s['pmid']}"
                meta = f"PMID {s['pmid']}"
                if s.get("year"):
                    meta += f" · {s['year']}"
                if s.get("study_type"):
                    meta += f" · {s['study_type']}"
                if s.get("journal"):
                    meta += f" · {s['journal']}"
                st.markdown(f"**{title}** `{meta}`")
            else:
                title = s.get("title") or s.get("policy_number") or "Unknown"
                st.markdown(f"**{title}** `{s.get('source', '')} {s.get('policy_number', '')}`")
            if excerpt := s.get("excerpt"):
                st.caption(excerpt)


def _source_meta(docs) -> list[dict]:
    return [
        {
            "title": s.metadata.get("title", ""),
            "policy_number": s.metadata.get("policy_number", ""),
            "source": s.metadata.get("source", ""),
            "excerpt": s.page_content,
        }
        for s in docs
    ]


def _pubmed_source_meta(docs) -> list[dict]:
    return [
        {
            "title": s.metadata.get("title", ""),
            "pmid": s.metadata.get("pmid", ""),
            "year": s.metadata.get("year", ""),
            "journal": s.metadata.get("journal", ""),
            "study_type": s.metadata.get("study_type", ""),
            "excerpt": s.page_content,
        }
        for s in docs
    ]


_MODE_TAG = {
    "gap": "Evidence Gap Analysis",
    "policy": "Policy Q&A",
    "coverage": "Coverage (NCD → LCD)",
}


def _render_trace(trace: list[dict]) -> None:
    """Show the agentic resolution path (which tools fired and what they returned)."""
    if not trace:
        return
    with st.expander("How this was resolved"):
        for step in trace:
            st.markdown(f"- `{step['tool']}` → **{step['result']}**")


def _handle_coverage(query: str, state: str | None = None) -> None:
    """Run the dynamic NCD→LCD coverage loop and render it, incl. the multi-turn state ask."""
    with st.spinner("Resolving coverage (NCD → LCD)..."):
        try:
            result = run_coverage_query(query, state=state)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    st.markdown(result["answer"])
    _render_trace(result.get("trace", []))
    # If the loop needs the beneficiary's state, remember the question so the next
    # message (the state) continues this same query.
    if result.get("needs_state"):
        st.session_state.pending_coverage = query
    new_index = len(st.session_state.messages)
    _render_feedback(new_index, query, mode="coverage")
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "trace": result.get("trace", [])}
    )
    _log_interaction(query, result["answer"], [], mode="coverage")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medicare Coverage Agent",
    page_icon="🏥",
    layout="wide",
)

st.title("Medicare Coverage Intelligence Agent")
st.caption(
    "Ask about Medicare coverage, request an evidence gap analysis, or check "
    "jurisdiction-specific coverage (mention a state for a live LCD lookup) — "
    "routing is automatic based on your question."
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "pending_coverage" not in st.session_state:
    st.session_state.pending_coverage = None

# ── Render history ────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user" and msg.get("mode"):
            st.caption(f"Routed to: {_MODE_TAG.get(msg['mode'], 'Policy Q&A')}")
        if msg.get("policy_sources"):
            _render_sources(msg["policy_sources"], "CMS Policy Sources")
        if msg.get("pubmed_sources"):
            _render_sources(msg["pubmed_sources"], "PubMed Evidence")
        if msg.get("sources"):
            _render_sources(msg["sources"])
        if msg.get("trace"):
            _render_trace(msg["trace"])
        if msg["role"] == "assistant":
            question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
            msg_mode = st.session_state.messages[i - 1].get("mode", "") if i > 0 else ""
            _render_feedback(i, question, mode=msg_mode)

# ── Handle new input ──────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a coverage question or request an evidence gap analysis..."):
    pending = st.session_state.pending_coverage
    st.session_state.pending_coverage = None
    if pending:
        # This message answers a prior "which state?" — continue that coverage query.
        mode, coverage_query = "coverage", pending
    else:
        mode, coverage_query = _route(prompt), prompt
    st.session_state.messages.append({"role": "user", "content": prompt, "mode": mode})

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"Routed to: {_MODE_TAG[mode]}")

    with st.chat_message("assistant"):
        if mode == "policy":
            with st.spinner("Retrieving policy documents..."):
                try:
                    result = answer(prompt)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            st.markdown(result["answer"])
            sources = _source_meta(result["sources"])
            _render_sources(sources)
            new_index = len(st.session_state.messages)
            _render_feedback(new_index, prompt, mode="policy")

            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": sources}
            )
            _log_interaction(prompt, result["answer"], result["sources"], mode="policy")

        elif mode == "coverage":
            _handle_coverage(coverage_query, state=prompt if pending else None)

        else:
            with st.spinner("Retrieving CMS policy and PubMed evidence..."):
                try:
                    result = gap_analysis(prompt)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            st.markdown(result["gap_report"])
            policy_sources = _source_meta(result["policy_sources"])
            pubmed_sources = _pubmed_source_meta(result["pubmed_sources"])
            _render_sources(policy_sources, "CMS Policy Sources")
            _render_sources(pubmed_sources, "PubMed Evidence")
            new_index = len(st.session_state.messages)
            _render_feedback(new_index, prompt, mode="gap")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["gap_report"],
                "policy_sources": policy_sources,
                "pubmed_sources": pubmed_sources,
            })
            _log_interaction(prompt, result["gap_report"], result["policy_sources"], mode="gap", pubmed_sources=result["pubmed_sources"])
