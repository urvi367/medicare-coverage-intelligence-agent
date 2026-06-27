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
import re
from dataclasses import dataclass

from src.ingestion.fetch import (
    _fetch_lcd_detail,
    _get_license_token,
    _paginate,
    _strip_html,
)

logger = logging.getLogger(__name__)

_LCD_LIST_ENDPOINT = "reports/local-coverage-final-lcds"
_TEXT_FIELDS = (
    "indications_limitations", "indications_limitations_text",
    "indication", "coverage_guidance", "summary_of_evidence",
)
# query/title noise to drop before token-overlap scoring
_STOP = {
    "for", "the", "of", "and", "in", "a", "an", "to", "is", "are", "does", "do",
    "covered", "cover", "coverage", "medicare", "evidence", "support", "supports",
    "patient", "patients", "use", "used", "using", "with", "this", "that", "service",
    "show", "what", "there", "any", "my", "their", "treatment", "therapy",
}

_list_cache: list[dict] | None = None


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


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP}


def _match_score(service_tokens: set[str], title: str) -> float:
    """Fraction of the service's content tokens present in the LCD title."""
    if not service_tokens:
        return 0.0
    return len(service_tokens & _tokens(title)) / len(service_tokens)


@dataclass
class LcdMatch:
    lcd_id: str        # display id, e.g. "L33610"
    title: str
    contractor: str    # first contractor line of contractor_name_type
    score: float
    text: str          # full indications/limitations text (HTML-stripped)


def lcd_lookup(
    mac: str,
    service: str,
    max_results: int = 3,
    min_score: float = 0.34,
) -> list[LcdMatch]:
    """Return up to `max_results` of this MAC's LCDs matching `service`, with full text.

    Args:
        mac: MAC contractor name (as in `contractor_name_type`), e.g. from resolve_mac().
        service: clean clinical-service description (e.g. "intravenous immune globulin").
        max_results: cap on LCDs returned.
        min_score: minimum title token-overlap (0–1) for a candidate to count.

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

    stoks = _tokens(service)
    ranked = sorted(
        ((_match_score(stoks, x.get("title", "")), x) for x in mac_items),
        key=lambda p: p[0],
        reverse=True,
    )
    top = [(s, x) for s, x in ranked if s >= min_score][:max_results]
    if not top:
        logger.info("No LCD title match for service %r under MAC %r", service, mac)
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
