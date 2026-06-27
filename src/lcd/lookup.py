"""Dynamic LCD lookup tool (Phase 3, step 2).

`lcd_lookup(mac, service)` resolves the relevant Local Coverage Determination(s)
for a MAC and a clinical service AT QUERY TIME — the part that cannot be sensibly
pre-indexed. Per the spike, the CMS list endpoint has no server-side filter, so:

    fetch the ~969-LCD list once (cached)         → cheap, refreshable
    → keep only this MAC's LCDs (substring on contractor_name_type)
    → rank by service/title token overlap, take the top matches
    → fetch full text on demand via the license-token-gated detail endpoint

Contract for the agentic loop:
    returns []                 → no LCD for this service in this jurisdiction
                                 (→ contractor discretion / individual consideration)
    returns [LcdMatch, ...]    → live LCD(s) with full text, ready to classify
    raises LcdLookupError      → the LCD *service* failed (list/token/network);
                                 the caller degrades gracefully, never fabricates
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from src.ingestion.fetch import (
    _fetch_lcd_detail,
    _get_license_token,
    _paginate,
    _strip_html,
)
from src.rag.pipeline import _get_reranker

logger = logging.getLogger(__name__)

_LCD_LIST_ENDPOINT = "reports/local-coverage-final-lcds"
_TEXT_FIELDS = (
    "indications_limitations", "indications_limitations_text",
    "indication", "coverage_guidance", "summary_of_evidence",
)
# bi-encoder cosine shortlist before cross-encoder scoring. Kept generous: the
# bi-encoder ranks synonym/terse-query matches surprisingly low (e.g. "panniculectomy"
# -> "Plastic Surgery" at cosine rank ~17), and the cross-encoder is the precision
# gate, so a larger shortlist improves recall without hurting precision.
_SHORTLIST_K = 30

_list_cache: list[dict] | None = None
_embedder = None
# per-MAC cache of (titles, normalized title-embedding matrix)
_title_emb_cache: dict[str, tuple[list[str], np.ndarray]] = {}


class LcdLookupError(RuntimeError):
    """The LCD service was unreachable or failed — the caller chooses the fallback."""


def _lcd_list() -> list[dict]:
    """Fetch (and cache for the process) the full final-LCD list."""
    global _list_cache
    if _list_cache is None:
        try:
            _list_cache = _paginate(_LCD_LIST_ENDPOINT)
        except Exception as e:  # network / HTTP / JSON
            raise LcdLookupError(f"LCD list fetch failed: {e}") from e
        logger.info("LCD list cached: %d records", len(_list_cache))
    return _list_cache


def _get_embedder():
    global _embedder
    if _embedder is None:
        from src.rag.embedder import get_embeddings
        _embedder = get_embeddings()
    return _embedder


def _rank_candidates(service: str, mac: str, items: list[dict]) -> list[tuple[float, dict]]:
    """Two-stage rank: bi-encoder cosine shortlist → cross-encoder relevance score.

    Returns (cross_encoder_sigmoid, item) pairs, highest first. The bi-encoder
    embeddings are normalized so a dot product is cosine; the cross-encoder then
    decides relevance (it separates on-topic from tangential far more cleanly than
    the bi-encoder's high cosine floor).
    """
    emb = _get_embedder()
    titles = [x.get("title", "") for x in items]
    key = f"{mac}|{len(titles)}"
    if key not in _title_emb_cache:
        _title_emb_cache[key] = (titles, np.asarray(emb.embed_documents(titles)))
    _, mat = _title_emb_cache[key]
    qv = np.asarray(emb.embed_query(service))
    cos = mat @ qv
    shortlist = np.argsort(-cos)[:_SHORTLIST_K]

    pairs = [(service, titles[i]) for i in shortlist]
    logits = _get_reranker().predict(pairs)
    scored = [(1.0 / (1.0 + math.exp(-float(l))), items[i]) for l, i in zip(logits, shortlist)]
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored


@dataclass
class LcdMatch:
    lcd_id: str        # display id, e.g. "L33610"
    title: str
    contractor: str    # first contractor line of contractor_name_type
    score: float       # cross-encoder relevance (sigmoid, 0–1)
    text: str          # full indications/limitations text (HTML-stripped)


def lcd_lookup(
    mac: str,
    service: str,
    max_results: int = 3,
    min_score: float = 0.575,
) -> list[LcdMatch]:
    """Return up to `max_results` of this MAC's LCDs matching `service`, with full text.

    Args:
        mac: MAC contractor name (as in `contractor_name_type`), e.g. from resolve_mac().
        service: clean clinical-service description (e.g. "intravenous immune globulin").
        max_results: cap on LCDs returned.
        min_score: minimum cross-encoder relevance (sigmoid, 0–1) for a candidate to
            count. 0.575 separates on-topic LCDs (≥0.597) from tangential ones (≤0.554)
            on the bge-reranker; a wrong LCD is worse than a "verify" fallback, so the
            threshold leans against false positives. Expects a clean service term.

    Returns [] when no LCD matches in this jurisdiction. Raises LcdLookupError on a
    list/token/network failure.
    """
    items = _lcd_list()
    # Filter to this MAC, deduping by display id (the list carries one row per
    # contractor/MAC-type, so the same LCD can appear several times) — keep the
    # highest document_version of each.
    by_id: dict[str, dict] = {}
    for x in items:
        if mac not in (x.get("contractor_name_type") or ""):
            continue
        did = x.get("document_display_id") or str(x.get("document_id"))
        if did not in by_id or (x.get("document_version", 0) or 0) > (by_id[did].get("document_version", 0) or 0):
            by_id[did] = x
    mac_items = list(by_id.values())
    if not mac_items:
        logger.info("No LCDs for MAC %r", mac)
        return []

    ranked = _rank_candidates(service, mac, mac_items)
    top = [(s, x) for s, x in ranked if s >= min_score][:max_results]
    if not top:
        logger.info("No relevant LCD for service %r under MAC %r (best %.3f)",
                    service, mac, ranked[0][0] if ranked else 0.0)
        return []

    try:
        token = _get_license_token()
    except Exception as e:
        raise LcdLookupError(f"LCD license token failed: {e}") from e

    matches: list[LcdMatch] = []
    for score, item in top:
        detail = _fetch_lcd_detail(item, token)  # returns item unchanged on detail failure
        text = next((_strip_html(str(detail[f])) for f in _TEXT_FIELDS if detail.get(f)), "")
        matches.append(LcdMatch(
            lcd_id=item.get("document_display_id", ""),
            title=item.get("title", ""),
            contractor=(item.get("contractor_name_type") or "").splitlines()[0].strip(),
            score=round(float(score), 3),
            text=text,
        ))
    return matches
