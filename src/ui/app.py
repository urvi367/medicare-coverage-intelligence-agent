"""Streamlit chat UI for the Medicare Coverage Intelligence Agent."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag.pipeline import answer

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


def _log_interaction(question: str, answer_text: str, sources: list) -> None:
    """Append one Q&A interaction to the JSONL log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
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
    """Append a feedback event to the JSONL log."""
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
    """Render feedback buttons for an assistant message, or its recorded rating."""
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


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            label = s.get("title") or s.get("policy_number") or "Unknown"
            st.markdown(f"**{label}** `{s.get('source','')} {s.get('policy_number','')}`")
            if excerpt := s.get("excerpt"):
                st.caption(excerpt)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medicare Coverage Agent",
    page_icon="🏥",
    layout="wide",
)

st.title("Medicare Coverage Intelligence Agent")
st.caption(
    "Ask questions about Medicare NCDs and LCDs. "
    "Answers are grounded in official CMS policy documents."
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ── Render history ────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        if msg["role"] == "assistant":
            # question is the preceding user message
            question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
            _render_feedback(i, question)

# ── Handle new input ──────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a Medicare coverage question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving policy documents..."):
            try:
                result = answer(prompt)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.markdown(result["answer"])

        source_meta = [
            {
                "title": s.metadata.get("title", ""),
                "policy_number": s.metadata.get("policy_number", ""),
                "source": s.metadata.get("source", ""),
                "excerpt": s.page_content[:400].strip(),
            }
            for s in result["sources"]
        ]
        _render_sources(source_meta)
        new_index = len(st.session_state.messages)  # user already appended; assistant will be at this index
        _render_feedback(new_index, prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": source_meta}
    )
    _log_interaction(prompt, result["answer"], result["sources"])
