"""Deterministic routing for dynamic LCD lookup: state -> MAC, and NCD disposition.

Scoped (Phase 3) to FIVE MAC jurisdictions, the best-represented in the data:
Noridian, CGS, WPS, Palmetto, National Government Services. States served by
First Coast (FL/PR/VI) and Novitas (AR/CO/LA/MS/NM/OK/TX/DE/DC/MD/NJ/PA) are
intentionally out of scope and resolve to None (the loop reports the
jurisdiction as unsupported rather than guessing).

Spike finding (CMS Coverage API): the LCD list endpoint
`reports/local-coverage-final-lcds` returns all ~969 final LCDs and IGNORES
server-side filters (keyword/state/contractor/q/search all no-op). So LCD
lookup = fetch the list once (cacheable) -> filter by `contractor_name_type`
locally -> fetch full text on demand via the license-token-gated detail
endpoint. There is no state field on LCDs, only the MAC contractor — hence
this external state->MAC table.

These routing decisions are deterministic by design; only the orchestration of
WHEN to call the LCD tools is agentic.
"""
from __future__ import annotations

import re

# Contractor names exactly as they appear in `contractor_name_type` in the LCD
# data, so the downstream filter is a direct string match.
NORIDIAN = "Noridian Healthcare Solutions, LLC"
CGS = "CGS Administrators, LLC"
WPS = "WPS Insurance Corporation"
PALMETTO = "Palmetto GBA"
NGS = "National Government Services, Inc."

SUPPORTED_MACS: tuple[str, ...] = (NORIDIAN, CGS, WPS, PALMETTO, NGS)

# Short slug per MAC for boolean index metadata (`mac_<key>: True`), so an LCD
# served by several MACs is stored ONCE with multiple flags — no duplication.
_MAC_KEYS: dict[str, str] = {
    NORIDIAN: "noridian", CGS: "cgs", WPS: "wps", PALMETTO: "palmetto", NGS: "ngs",
}


def mac_key(mac: str) -> str:
    """Slug for a MAC contractor name, used as the `mac_<key>` metadata flag."""
    return _MAC_KEYS.get(mac, "")

# State / territory (USPS code) -> MAC contractor, for the 5 in-scope MACs only.
STATE_TO_MAC: dict[str, str] = {
    # Noridian — JE + JF
    **{s: NORIDIAN for s in ("AK", "AZ", "CA", "HI", "ID", "MT", "ND", "NV",
                              "OR", "SD", "UT", "WA", "WY", "AS", "GU", "MP")},
    # CGS — J15
    **{s: CGS for s in ("KY", "OH")},
    # WPS — J5 + J8
    **{s: WPS for s in ("IA", "IN", "KS", "MI", "MO", "NE")},
    # Palmetto — JJ + JM
    **{s: PALMETTO for s in ("AL", "GA", "NC", "SC", "TN", "VA", "WV")},
    # National Government Services — J6 + JK
    **{s: NGS for s in ("CT", "IL", "MA", "ME", "MN", "NH", "NY", "RI", "VT", "WI")},
}

# Full state names -> USPS code, for extracting a state from free-text queries.
_STATE_NAMES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    "puerto rico": "PR",
}
_VALID_CODES = set(_STATE_NAMES.values())
# whole-word USPS code matcher (avoids matching "IN"/"OR"/"OH" inside words)
_CODE_RE = re.compile(r"\b([A-Z]{2})\b")


def extract_state(query: str) -> str | None:
    """Best-effort extract a USPS state code from a free-text query, else None.

    Tries full state names first (unambiguous), then explicit 2-letter codes.
    Returns None when no state is present — the agentic loop uses that to decide
    whether to ASK the user for their state before resolving a MAC.
    """
    low = query.lower()
    for name, code in _STATE_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            return code
    for m in _CODE_RE.findall(query):
        if m in _VALID_CODES:
            return m
    return None


def resolve_mac(state: str | None) -> str | None:
    """Map a USPS state code to its MAC contractor name (matching the LCD data).

    Returns None when state is missing, unknown, or served by an out-of-scope MAC
    (First Coast / Novitas) — the caller reports it rather than guessing.
    """
    if not state:
        return None
    return STATE_TO_MAC.get(state.strip().upper())


# ── NCD disposition ──────────────────────────────────────────────────────────
# High-precision markers that an NCD has NO national determination / leaves the
# decision to the local MAC. Kept narrow on purpose: a passing mention of "MAC"
# (e.g. procedural notes) is NOT a defer — only an explicit hand-off is.
_DEFER_RE = re.compile(
    r"no national coverage determination"
    r"|there is no ncd\b"
    r"|coverage determinations?\s+(?:will be|are)\s+made by the (?:local )?medicare administrative contractor"
    r"|at the discretion of the (?:local )?medicare administrative contractor"
    r"|left to the discretion of the (?:local|medicare administrative contractor)"
    r"|determined by the local medicare",
    re.I,
)


def ncd_disposition(ncd_text: str | None) -> str:
    """Classify how the (deterministically selected) NCD treats a service.

    Returns:
        "silent"  — no governing NCD was found at all (escalate to LCD),
        "defers"  — an NCD exists but explicitly hands coverage to the local MAC
                    (escalate to LCD),
        "governs" — the NCD makes a national determination (use it; no LCD needed).
    """
    if not ncd_text or not ncd_text.strip():
        return "silent"
    return "defers" if _DEFER_RE.search(ncd_text) else "governs"
