"""Fetch NCDs and LCDs from the CMS Coverage API (https://api.coverage.cms.gov/v1)."""

import concurrent.futures
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE = "https://api.coverage.cms.gov/v1"
DATA_DIR = Path(__file__).parents[2] / "data"
PAGE_SIZE = 100

_session = requests.Session()


def _strip_html(text: str) -> str:
    """Remove HTML tags and fully decode entities from a string.

    CMS data is frequently double-escaped (e.g. '&amp;gt;' is the literal text for
    '>'). A single html.unescape only peels one layer, leaving a stray '&gt;', so we
    unescape repeatedly until stable before stripping the (now literal) tags.
    """
    for _ in range(5):                     # peel nested escaping: &amp;gt; → &gt; → >
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    # Strip real HTML tags only — must start with a letter or '/'. This deliberately
    # spares clinical comparisons like '< 80 mm Hg' / '> 130' that decoding exposed.
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _paginate(endpoint: str, headers: dict | None = None) -> list[dict[str, Any]]:
    """Collect all pages from a list endpoint using next_token pagination."""
    records: list[dict[str, Any]] = []
    next_token = ""

    while True:
        params: dict[str, Any] = {"limit": PAGE_SIZE}
        if next_token:
            params["next_token"] = next_token

        resp = _session.get(
            f"{BASE}/{endpoint}",
            params=params,
            headers=headers or {},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()

        page: list = body.get("data", [])
        if not page:
            break

        records.extend(page)
        next_token = body.get("meta", {}).get("next_token", "")
        if not next_token or len(page) < PAGE_SIZE:
            break

    return records


def _get_license_token() -> str:
    """Fetch a one-hour Bearer token required by the LCD detail endpoint."""
    resp = _session.get(f"{BASE}/metadata/license-agreement", timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["Token"]


def _fetch_ncd_detail(item: dict[str, Any]) -> dict[str, Any]:
    """Fetch full policy text for one NCD and merge with its list record."""
    ncd_id = item["document_id"]
    ncd_ver = item["document_version"]
    try:
        resp = _session.get(
            f"{BASE}/data/ncd",
            params={"ncdid": ncd_id, "ncdver": ncd_ver},
            timeout=30,
        )
        resp.raise_for_status()
        detail = resp.json().get("data", [{}])[0]
        return {**item, **detail}
    except Exception as e:
        logger.warning("NCD %s detail failed: %s", ncd_id, e)
        return item


def _fetch_lcd_detail(item: dict[str, Any], token: str) -> dict[str, Any]:
    """Fetch full policy text for one LCD using the license Bearer token."""
    lcd_id = item["document_id"]
    lcd_ver = item["document_version"]
    try:
        resp = _session.get(
            f"{BASE}/data/lcd",
            params={"lcdid": lcd_id, "lcdver": lcd_ver},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        detail = resp.json().get("data", [{}])[0]
        return {**item, **detail}
    except Exception as e:
        logger.warning("LCD %s detail failed: %s", lcd_id, e)
        return item


# ── LCD diagnostic-test classification (for evidence-search anchoring) ──────────
# LCDs carry no benefit_category (unlike NCDs), so a diagnostic/lab-test LCD is
# identified by its CPT/HCPCS codes — which live in the associated Billing & Coding
# ARTICLE, not the LCD (lcd/hcpc-code is empty). Chain: lcd/related-documents →
# article/hcpc-code. A policy is a diagnostic test when every numeric CPT falls in a
# diagnostic range and it carries no drug (J/Q) code. Ranges validated on the golden
# LCD set: flags magnesium/HbA1c/SSEP + other real tests, excludes every therapy.
_DIAGNOSTIC_CPT_RANGES = [
    (70010, 76999),  # diagnostic radiology (77xxx radiation therapy excluded)
    (78012, 78999),  # diagnostic nuclear medicine (79xxx therapy excluded)
    (80047, 89398),  # pathology & laboratory
    (92002, 92287),  # ophthalmology diagnostic (92310+ lenses/therapy excluded)
    (93000, 93356),  # cardiovascular diagnostic (93797+ cardiac rehab excluded)
    (94002, 94799),  # pulmonary function testing
    (95700, 96020),  # neuro diagnostic: EEG/EMG/evoked potentials/sleep/autonomic
]


def _is_diagnostic_hcpc(codes: list[str]) -> bool:
    """True when a code set marks a pure diagnostic/lab test.

    Requires >=1 numeric 5-digit CPT, ALL numeric CPTs in a diagnostic range, and no
    HCPCS drug code (J/Q, = pharmacological therapy). Any procedure/therapy CPT (e.g.
    64450 nerve block, 97xxx PT) fails the all-diagnostic test, keeping mixed policies out.
    """
    nums: list[int] = []
    for c in codes:
        c = str(c).strip().upper()
        if c[:1] in ("J", "Q"):          # HCPCS drug code → pharmacological therapy
            return False
        if c.isdigit() and len(c) == 5:
            nums.append(int(c))
    if not nums:
        return False
    return all(any(lo <= n <= hi for lo, hi in _DIAGNOSTIC_CPT_RANGES) for n in nums)


def _lcd_hcpc_codes(doc_id: int, ver: int, token: str) -> list[str]:
    """All CPT/HCPCS codes for an LCD via its related Billing & Coding article(s).

    CMS keeps coding on the article, not the LCD, so follow lcd/related-documents →
    article/hcpc-code, unioning codes across the related articles.
    """
    H = {"Authorization": f"Bearer {token}"}
    r = _session.get(f"{BASE}/data/lcd/related-documents",
                     params={"lcdid": doc_id, "ver": ver}, headers=H, timeout=30)
    r.raise_for_status()
    arts = [(row["r_article_id"], row["r_article_version"])
            for row in r.json().get("data", []) if row.get("r_article_id")]
    codes: set[str] = set()
    for aid, aver in arts:
        rr = _session.get(f"{BASE}/data/article/hcpc-code",
                          params={"articleid": aid, "ver": aver}, headers=H, timeout=30)
        if rr.status_code == 200:
            codes.update(str(x["hcpc_code_id"]) for x in rr.json().get("data", []))
    return sorted(codes)


def build_lcd_diagnostic_map() -> Path:
    """Classify every active LCD as diagnostic-test or not, from its article CPT codes.

    Saves data/lcd_diagnostic.json ({display_id: {diagnostic, codes}}). Resumable;
    consumed by fetch_lcd_evidence to anchor diagnostic LCDs' PubMed search on the test.
    """
    import time
    lcds = json.loads((DATA_DIR / "lcd_raw.json").read_text(encoding="utf-8"))
    active = [r for r in lcds if r.get("document_display_id")
              and (r.get("retirement_date") or "N/A").strip() == "N/A"]
    out_path = DATA_DIR / "lcd_diagnostic.json"
    result: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    logger.info("Classifying %d active LCDs (%d already cached)...", len(active), len(result))

    token = _get_license_token()
    for i, r in enumerate(active, 1):
        did = r["document_display_id"]
        if did in result:
            continue
        try:
            codes = _lcd_hcpc_codes(r["document_id"], r["document_version"], token)
        except Exception as e:
            logger.warning("LCD %s codes failed: %s", did, e)
            continue
        result[did] = {"diagnostic": _is_diagnostic_hcpc(codes), "codes": codes}
        if i % 25 == 0:
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info("  ...%d/%d (%d diagnostic so far)", i, len(active),
                        sum(1 for v in result.values() if v["diagnostic"]))
        time.sleep(0.15)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    ndiag = sum(1 for v in result.values() if v["diagnostic"])
    logger.info("Saved %d LCD classifications (%d diagnostic) → %s", len(result), ndiag, out_path)
    return out_path


def fetch_and_save(doc_type: str, max_docs: int = 500) -> Path:
    """Fetch doc_type ('ncd' or 'lcd') list + full text and save to data/{doc_type}_raw.json.

    Args:
        doc_type: 'ncd' or 'lcd'
        max_docs: Cap on number of documents to fetch (default 500).
    """
    if doc_type == "ncd":
        logger.info("Fetching NCD list...")
        items = _paginate("reports/national-coverage-ncd")[:max_docs]
        logger.info("  → %d NCDs, fetching full text in parallel...", len(items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(_fetch_ncd_detail, items))

    else:
        logger.info("Obtaining LCD license token...")
        try:
            token = _get_license_token()
            logger.info("  → token obtained")
        except Exception as e:
            logger.warning("License token failed (%s) — saving list metadata only", e)
            token = ""

        logger.info("Fetching LCD list...")
        items = _paginate("reports/local-coverage-final-lcds")[:max_docs]
        logger.info("  → %d LCDs, fetching full text in parallel...", len(items))

        if token:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                records = list(
                    pool.map(lambda item: _fetch_lcd_detail(item, token), items)
                )
        else:
            records = items

    out = DATA_DIR / f"{doc_type}_raw.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved %d records to %s", len(records), out)
    return out


def fetch_lcds_for_macs(macs: list[str]) -> Path:
    """Fetch ACTIVE LCDs (with full body text) for the given MAC contractors.

    Re-fetches via the live list + license-token detail endpoint (the body lives in
    the detail response, not the list). Filters out retired LCDs and any not served
    by an in-scope MAC. Saves to data/lcd_raw.json, replacing the stale snapshot.
    """
    logger.info("Obtaining LCD license token...")
    token = _get_license_token()
    items = _paginate("reports/local-coverage-final-lcds")
    selected = [
        x for x in items
        if (x.get("retirement_date") or "N/A").strip() == "N/A"
        and any(m in (x.get("contractor_name_type") or "") for m in macs)
    ]
    logger.info("  → %d active in-scope LCDs; fetching full text in parallel...", len(selected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda it: _fetch_lcd_detail(it, token), selected))

    # Keep only fields used for indexing/metadata — drop bulky unused sections
    # (bibliography, summary_of_evidence, …) so the saved file stays small.
    keep = {"document_id", "document_version", "document_display_id", "title",
            "contractor_name_type", "retirement_date", "indication",
            "indications_limitations", "indications_limitations_text",
            "coverage_indications", "description", "cms_cov_policy"}
    records = [{k: v for k, v in r.items() if k in keep} for r in records]

    out = DATA_DIR / "lcd_raw.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved %d LCDs (with body) to %s", len(records), out)
    return out


def load_documents(doc_type: str) -> list[dict[str, str]]:
    """Load saved raw JSON and return cleaned text + metadata dicts ready for indexing."""
    path = DATA_DIR / f"{doc_type}_raw.json"
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    # NCD detail fields confirmed from API; LCD fields may vary, so we try several
    ncd_text_fields = ["item_service_description", "indications_limitations"]
    # `indication` is the LCD's indications/limitations-of-coverage section (the core
    # coverage criteria); the others are fallbacks for differing API shapes.
    lcd_text_fields = ["indication", "indications_limitations", "indications_limitations_text",
                       "coverage_indications", "description"]
    text_fields = ncd_text_fields if doc_type == "ncd" else lcd_text_fields

    docs: list[dict[str, str]] = []
    for r in records:
        body = "\n\n".join(r.get(f, "") for f in text_fields if r.get(f))
        title = r.get("title", "")
        text = _strip_html(f"{title}\n\n{body}".strip())
        if not text:
            continue
        base = {
            "text": text,
            "source": doc_type.upper(),
            "title": title,
            "policy_number": r.get("document_display_id", ""),
            "doc_id": str(r.get("document_id", "")),
        }
        if doc_type != "lcd":
            docs.append(base)
            continue
        # LCDs are jurisdictional: skip retired, and tag with the serving in-scope
        # MAC(s) as boolean `mac_<key>` flags — ONE doc per LCD even when shared across
        # MACs (no duplication; Chroma metadata is scalar so a list won't filter).
        if (r.get("retirement_date") or "N/A").strip() != "N/A":
            continue
        # lazy import: ingestion → pipeline at module top would be circular
        from src.rag.pipeline import SUPPORTED_MACS, mac_key
        contractor = r.get("contractor_name_type", "") or ""
        serving = [m for m in SUPPORTED_MACS if m in contractor]
        if not serving:
            continue
        base["contractor"] = contractor.splitlines()[0].strip()
        for m in serving:
            base[f"mac_{mac_key(m)}"] = True
        docs.append(base)

    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_and_save("ncd")
    fetch_and_save("lcd")
