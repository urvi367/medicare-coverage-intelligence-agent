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


def load_documents(doc_type: str) -> list[dict[str, str]]:
    """Load saved raw JSON and return cleaned text + metadata dicts ready for indexing."""
    path = DATA_DIR / f"{doc_type}_raw.json"
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    # NCD detail fields confirmed from API; LCD fields may vary, so we try several
    ncd_text_fields = ["item_service_description", "indications_limitations"]
    lcd_text_fields = ["indications_limitations", "indications_limitations_text",
                       "coverage_indications", "description"]
    text_fields = ncd_text_fields if doc_type == "ncd" else lcd_text_fields

    docs: list[dict[str, str]] = []
    for r in records:
        body = "\n\n".join(r.get(f, "") for f in text_fields if r.get(f))
        title = r.get("title", "")
        text = _strip_html(f"{title}\n\n{body}".strip())
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "source": doc_type.upper(),
                "title": title,
                "policy_number": r.get("document_display_id", ""),
                "doc_id": str(r.get("document_id", "")),
            }
        )

    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_and_save("ncd")
    fetch_and_save("lcd")
