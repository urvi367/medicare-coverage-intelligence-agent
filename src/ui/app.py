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


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            label = s.get("title") or s.get("policy_number") or "Unknown"
            st.markdown(f"- **{label}** `{s.get('source','')} {s.get('policy_number','')}`")


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

# ── Render history ────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

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
            }
            for s in result["sources"]
        ]
        _render_sources(source_meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": source_meta}
    )
    _log_interaction(prompt, result["answer"], result["sources"])
