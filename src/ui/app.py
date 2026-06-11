"""Streamlit chat UI for the Medicare Coverage Intelligence Agent."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag.pipeline import answer, gap_analysis

load_dotenv()
logging.basicConfig(level=logging.INFO)

LOG_DIR = Path(__file__).parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "interactions.jsonl"


_FEEDBACK_LABELS = {
    "positive": "✅ Yes, this answered my question",
    "partial":  "⚠️ Partially — missing some details",
    "negative": "❌ No, this seems incorrect",
}


def _log_interaction(question: str, answer_text: str, sources: list, mode: str) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "question": question,
        "answer": answer_text,
        "sources": [
            {
                "title": s.metadata.get("title", ""),
                "policy_number": s.metadata.get("policy_number", ""),
                "source": s.metadata.get("source", ""),
            }
            for s in sources
        ],
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _log_feedback(question: str, rating: str) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "feedback",
        "question": question,
        "rating": rating,
        "label": _FEEDBACK_LABELS[rating],
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _render_feedback(msg_index: int, question: str) -> None:
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if msg_index in st.session_state.feedback:
        st.caption(f"Your feedback: {_FEEDBACK_LABELS[st.session_state.feedback[msg_index]]}")
        return
    cols = st.columns(3)
    for col, (key, label) in zip(cols, _FEEDBACK_LABELS.items()):
        if col.button(label, key=f"fb_{msg_index}_{key}"):
            st.session_state.feedback[msg_index] = key
            _log_feedback(question, key)
            st.rerun()


def _render_sources(sources: list[dict], label: str = "Sources") -> None:
    if not sources:
        return
    with st.expander(label):
        for s in sources:
            title = s.get("title") or s.get("policy_number") or "Unknown"
            st.markdown(f"**{title}** `{s.get('source','')} {s.get('policy_number','')}`")
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


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medicare Coverage Agent",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar mode selector ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode for your next question:",
        ["Policy Q&A", "Evidence Gap Analysis"],
        key="mode_selector",
    )
    st.divider()
    if mode == "Policy Q&A":
        st.caption("Answers coverage questions using CMS NCDs and LCDs.")
    else:
        st.caption("Compares CMS NCD coverage criteria against published PubMed clinical evidence.")

# ── Main header ───────────────────────────────────────────────────────────────

st.title("Medicare Coverage Intelligence Agent")
st.caption("Ask coverage questions and evidence gap questions in any order — switch modes in the sidebar.")

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ── Render history ────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            mode_tag = msg.get("mode")
            if mode_tag:
                label = "Policy Q&A" if mode_tag == "policy" else "Evidence Gap Analysis"
                st.caption(f"Mode: {label}")
        else:
            st.markdown(msg["content"])
            if msg.get("policy_sources"):
                _render_sources(msg["policy_sources"], "CMS Policy Sources")
            if msg.get("pubmed_sources"):
                _render_sources(msg["pubmed_sources"], "PubMed Evidence")
            if msg.get("sources"):
                _render_sources(msg["sources"])
            question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
            _render_feedback(i, question)

# ── Handle new input ──────────────────────────────────────────────────────────

placeholder = (
    "Ask a Medicare coverage question..."
    if mode == "Policy Q&A"
    else "Enter a clinical topic to compare CMS policy vs evidence..."
)

if prompt := st.chat_input(placeholder):
    mode_key = "policy" if mode == "Policy Q&A" else "gap"
    st.session_state.messages.append({"role": "user", "content": prompt, "mode": mode_key})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"Mode: {mode}")

    with st.chat_message("assistant"):
        if mode_key == "policy":
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
            _render_feedback(new_index, prompt)

            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": sources}
            )
            _log_interaction(prompt, result["answer"], result["sources"], mode="policy")

        else:
            with st.spinner("Retrieving CMS policy and PubMed evidence..."):
                try:
                    result = gap_analysis(prompt)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            st.markdown(result["gap_report"])
            policy_sources = _source_meta(result["policy_sources"])
            pubmed_sources = _source_meta(result["pubmed_sources"])
            _render_sources(policy_sources, "CMS Policy Sources")
            _render_sources(pubmed_sources, "PubMed Evidence")
            new_index = len(st.session_state.messages)
            _render_feedback(new_index, prompt)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["gap_report"],
                "policy_sources": policy_sources,
                "pubmed_sources": pubmed_sources,
            })
            _log_interaction(prompt, result["gap_report"], result["policy_sources"], mode="gap")
