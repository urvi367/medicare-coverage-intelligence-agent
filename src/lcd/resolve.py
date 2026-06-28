"""Governing-policy cascade — shared by Policy Q&A and Gap Analysis (Phase 3).

NCD first: if a National Coverage Determination governs the service, that IS the
answer. Otherwise fall back to the beneficiary's jurisdiction LCD (fetched live).
If neither governs, nothing does ("idk"). This is the resolution step both
`answer()` and `gap_analysis()` call — not a separate path. Routing is
deterministic (relevance gate + defer-marker + state→MAC); only the live LCD
fetch is dynamic tool-use.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.lcd.jurisdiction import _STATE_NAMES, extract_state, ncd_disposition, resolve_mac
from src.lcd.lookup import LcdLookupError, lcd_lookup

logger = logging.getLogger(__name__)

# Top-NCD rerank sigmoid below this ⇒ no NCD is on-topic ⇒ go local (governed
# services score ~0.73, LCD-only ~0.55 on the golden set).
SILENT_GATE = 0.60

# Coverage-question framing to strip so the LCD matcher sees the clinical service,
# not "is … covered … in <state>?" noise (the matcher expects a clean service term).
_QUESTION_NOISE = re.compile(
    r"\b(is|are|was|were|does|do|did|can|could|will|would|should|has|have|"
    r"cover|covered|coverage|covers|for|my|the|a|an|to|in|of|on|under|when|"
    r"whether|if|patient|patients|beneficiary|beneficiaries|medicare|cms|"
    r"reimburse|reimbursed|reimbursement|eligible|service|please|this|that|"
    r"there|any|state|jurisdiction|region|area|local|lcd|ncd)\b",
    re.IGNORECASE,
)


def _service_term(question: str, state_code: str) -> str:
    """Strip the state mention + coverage-question framing → a clean service term."""
    s = question
    for name, code in _STATE_NAMES.items():
        if code == state_code:
            s = re.sub(rf"\b{re.escape(name)}\b", " ", s, flags=re.IGNORECASE)
    if state_code:
        s = re.sub(rf"\b{re.escape(state_code)}\b", " ", s)
    s = _QUESTION_NOISE.sub(" ", s)
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
    # for source == "none": why — need_state | unsupported_jurisdiction
    #                              | lookup_failed | no_determination
    note: str = ""


def resolve_governing_policy(
    question: str, state: str | None = None, disambiguate: bool = False
) -> ResolvedPolicy:
    """Resolve the one governing policy via the NCD→LCD cascade.

    Args:
        question: the coverage question.
        state: optional known state (e.g. from a prior "which state?" turn).
        disambiguate: if True, use the LLM disambiguation step to pick the primary
            NCD (gap analysis wants this); if False, the deterministic vote (cheaper,
            for Policy Q&A).
    """
    # Lazy import to avoid a circular dependency (pipeline imports this lazily too).
    from src.rag.pipeline import (
        PIPELINE_CONFIG,
        _full_ncd_docs,
        _hybrid_retrieve_ncd,
        _rerank_scored,
        _select_primary_ncds,
    )

    # ── NCD first ────────────────────────────────────────────────────────────
    reranked = _rerank_scored(
        question, _hybrid_retrieve_ncd(question, PIPELINE_CONFIG["k"]),
        top_n=PIPELINE_CONFIG["reranker_top_n"],
    )
    top_sig = 1.0 / (1.0 + math.exp(-reranked[0][0])) if reranked else 0.0
    if reranked and top_sig >= SILENT_GATE:
        primary = _select_primary_ncds(reranked, question=question if disambiguate else None)
        if primary:
            docs = [d for n in primary for d in _full_ncd_docs(n)]
            if ncd_disposition("\n".join(d.page_content for d in docs)) == "governs":
                return ResolvedPolicy(
                    source="ncd", policy_docs=docs, policy_ids=list(primary),
                    title=docs[0].metadata.get("title", "") if docs else "",
                )
            # NCD exists but defers to local contractors → fall through to LCD

    # ── LCD fallback (jurisdictional, live) ──────────────────────────────────
    code = extract_state(question) or extract_state(state or "") \
        or (state.strip().upper() if state else "")
    if not code:
        return ResolvedPolicy(source="none", needs_state=True, note="need_state")
    mac = resolve_mac(code)
    if not mac:
        return ResolvedPolicy(source="none", note="unsupported_jurisdiction")
    try:
        matches = lcd_lookup(mac, _service_term(question, code))
    except LcdLookupError as e:
        logger.warning("LCD lookup failed: %s", e)
        return ResolvedPolicy(source="none", mac=mac, note="lookup_failed")
    if not matches:
        return ResolvedPolicy(source="none", mac=mac, note="no_determination")

    lcd_docs = [
        Document(page_content=m.text, metadata={
            "source": "LCD", "policy_number": m.lcd_id, "title": m.title,
            "contractor": m.contractor,
        })
        for m in matches
    ]
    return ResolvedPolicy(
        source="lcd", policy_docs=lcd_docs, policy_ids=[m.lcd_id for m in matches],
        title=matches[0].title, mac=mac,
    )
