"""Governing-policy cascade — shared by Policy Q&A and Gap Analysis (Phase 3).

NCD first: if a National Coverage Determination governs the service, that IS the
answer. Otherwise fall back to the beneficiary's jurisdiction LCD, retrieved by
RAG over the indexed LCD bodies and filtered to that MAC. If neither governs,
nothing does ("idk"). This is the resolution step both `answer()` and
`gap_analysis()` call — not a separate path. Routing is deterministic (relevance
gates + defer-marker + state→MAC); retrieval is body-based RAG, like NCDs.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.lcd.jurisdiction import _STATE_NAMES, extract_state, ncd_disposition, resolve_mac

logger = logging.getLogger(__name__)

# Top reranked-chunk sigmoid below this ⇒ nothing on-topic governs.
SILENT_GATE = 0.60   # NCD side (governed ~0.73 vs LCD-only ~0.55)
LCD_GATE = 0.55      # LCD side: no MAC LCD is relevant enough → contractor discretion

# Coverage/evidence-question framing stripped before LCD retrieval: LCD bodies share
# a lot of "covered…patient…medically necessary" boilerplate, so a noisy query matches
# the wrong LCD — denoising to the clinical service term fixes retrieval (NOT title
# matching; the match is still over the body).
_QNOISE = re.compile(
    r"\b(is|are|was|were|does|do|did|can|could|will|would|should|has|have|cover|covered|"
    r"coverage|covers|for|my|the|a|an|to|in|of|on|under|when|whether|if|patient|patients|"
    r"beneficiary|medicare|cms|reimburse|reimbursed|eligible|service|please|this|that|"
    r"there|any|state|jurisdiction|region|area|local|lcd|ncd|what|whats|evidence|show|"
    r"shows|support|supports|supported|about|data|research|study|studies|literature|"
    r"clinical|effective|effectiveness|patient's)\b",
    re.IGNORECASE,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _denoise(question: str, state_code: str) -> str:
    """Strip the state mention + coverage/evidence-question framing → clinical term."""
    s = question
    for name, code in _STATE_NAMES.items():
        if code == state_code:
            s = re.sub(rf"\b{re.escape(name)}\b", " ", s, flags=re.IGNORECASE)
    if state_code:
        s = re.sub(rf"\b{re.escape(state_code)}\b", " ", s)
    s = _QNOISE.sub(" ", s)
    s = re.sub(r"[^\w\s./-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or question


@dataclass
class ResolvedPolicy:
    """The single governing policy for a question (or none)."""
    source: str                                   # "ncd" | "lcd" | "none"
    policy_docs: list[Document] = field(default_factory=list)
    policy_ids: list[str] = field(default_factory=list)
    title: str = ""
    mac: str | None = None
    needs_state: bool = False
    # for source == "none": why — need_state | unsupported_jurisdiction | no_determination
    note: str = ""


def resolve_governing_policy(
    question: str, state: str | None = None, disambiguate: bool = False
) -> ResolvedPolicy:
    """Resolve the one governing policy via the NCD→LCD cascade.

    Args:
        question: the coverage question.
        state: optional known state (e.g. from a prior "which state?" turn).
        disambiguate: use the LLM disambiguation step to pick the primary NCD (gap
            analysis), vs the deterministic vote (Policy Q&A).
    """
    # Lazy import to avoid a circular dependency (pipeline imports this lazily too).
    from src.rag.pipeline import (
        PIPELINE_CONFIG,
        _full_lcd_docs,
        _full_ncd_docs,
        _hybrid_retrieve_lcd,
        _hybrid_retrieve_ncd,
        _rerank_scored,
        _select_primary_ncds,
    )
    top_n = PIPELINE_CONFIG["reranker_top_n"]

    # ── NCD first ────────────────────────────────────────────────────────────
    reranked = _rerank_scored(question, _hybrid_retrieve_ncd(question, PIPELINE_CONFIG["k"]), top_n=top_n)
    if reranked and _sigmoid(reranked[0][0]) >= SILENT_GATE:
        primary = _select_primary_ncds(reranked, question=question if disambiguate else None)
        if primary:
            docs = [d for n in primary for d in _full_ncd_docs(n)]
            if ncd_disposition("\n".join(d.page_content for d in docs)) == "governs":
                return ResolvedPolicy(
                    source="ncd", policy_docs=docs, policy_ids=list(primary),
                    title=docs[0].metadata.get("title", "") if docs else "",
                )
            # NCD exists but defers to local contractors → fall through to LCD

    # ── LCD fallback (jurisdictional, RAG over indexed LCD bodies) ────────────
    code = extract_state(question) or extract_state(state or "") \
        or (state.strip().upper() if state else "")
    if not code:
        return ResolvedPolicy(source="none", needs_state=True, note="need_state")
    mac = resolve_mac(code)
    if not mac:
        return ResolvedPolicy(source="none", note="unsupported_jurisdiction")

    lcd_q = _denoise(question, code)
    lcd_ranked = _rerank_scored(
        lcd_q, _hybrid_retrieve_lcd(lcd_q, PIPELINE_CONFIG["k"], mac), top_n=top_n)
    if not lcd_ranked or _sigmoid(lcd_ranked[0][0]) < LCD_GATE:
        return ResolvedPolicy(source="none", mac=mac, note="no_determination")

    top_lcd = lcd_ranked[0][1].metadata.get("policy_number", "")
    docs = _full_lcd_docs(top_lcd, mac)  # whole-LCD context for the governing LCD
    return ResolvedPolicy(
        source="lcd", policy_docs=docs, policy_ids=[top_lcd],
        title=docs[0].metadata.get("title", "") if docs else "", mac=mac,
    )
