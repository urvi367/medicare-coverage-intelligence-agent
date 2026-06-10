"""Fetch PubMed abstracts for each NCD topic via NCBI E-utilities."""

import json
import logging
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_DELAY = 0.34  # 3 req/s — NCBI unauthenticated rate limit

_session = requests.Session()


def _base_params() -> dict:
    return {"db": "pubmed", "retmode": "json"}


def _search_pmids(query: str, max_results: int = 10) -> list[str]:
    """Return up to max_results PMIDs for query, sorted by relevance."""
    params = {**_base_params(), "term": query, "retmax": max_results, "sort": "relevance"}
    try:
        r = _session.get(ESEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.json()["esearchresult"].get("idlist", [])
    except Exception as e:
        logger.warning("esearch failed for %r: %s", query, e)
        return []


def _fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch PubMed XML for a list of PMIDs and parse into dicts."""
    if not pmids:
        return []
    params = {**_base_params(), "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    params.pop("retmode", None)  # efetch uses rettype, not retmode for XML
    try:
        r = _session.get(EFETCH_URL, params=params, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logger.warning("efetch failed for pmids %s: %s", pmids[:3], e)
        return []

    records = []
    try:
        root = ElementTree.fromstring(r.content)
    except ElementTree.ParseError as e:
        logger.warning("XML parse error: %s", e)
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        # AbstractText can have multiple sections (structured abstract)
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(el.itertext()) for el in abstract_parts).strip()

        year_el = (
            article.find(".//PubDate/Year")
            or article.find(".//PubDate/MedlineDate")
        )
        year = year_el.text[:4] if year_el is not None and year_el.text else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        if abstract:
            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
            })
    return records


def fetch_and_save(max_per_topic: int = 10) -> Path:
    """Search PubMed for each NCD title, fetch abstracts, and save to data/pubmed_raw.json.

    Args:
        max_per_topic: Max abstracts to fetch per NCD topic (default 10).

    Returns:
        Path to the saved JSON file.
    """
    ncd_path = DATA_DIR / "ncd_raw.json"
    ncd_records = json.loads(ncd_path.read_text(encoding="utf-8"))
    topics = [
        {"title": r["title"], "policy_number": r.get("document_display_id", "")}
        for r in ncd_records
        if r.get("title")
    ]
    logger.info("Fetching PubMed abstracts for %d NCD topics (max %d each)...", len(topics), max_per_topic)

    seen_pmids: set[str] = set()
    all_records: list[dict] = []

    for i, topic in enumerate(topics, 1):
        query = topic["title"]
        logger.info("  [%d/%d] %s", i, len(topics), query)

        time.sleep(_DELAY)
        pmids = _search_pmids(query, max_per_topic)
        if not pmids:
            continue

        time.sleep(_DELAY)
        abstracts = _fetch_abstracts(pmids)

        for rec in abstracts:
            if rec["pmid"] in seen_pmids:
                continue
            seen_pmids.add(rec["pmid"])
            all_records.append({
                **rec,
                "source_ncd_title": topic["title"],
                "source_ncd_number": topic["policy_number"],
            })

    out = DATA_DIR / "pubmed_raw.json"
    out.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    logger.info("Saved %d unique abstracts to %s", len(all_records), out)
    return out


def load_documents() -> list[dict[str, str]]:
    """Load saved PubMed abstracts as indexable dicts (text + metadata)."""
    path = DATA_DIR / "pubmed_raw.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for r in records:
        text = f"{r['title']}\n\n{r['abstract']}".strip()
        if not text:
            continue
        docs.append({
            "text": text,
            "source": "PUBMED",
            "pmid": r["pmid"],
            "title": r["title"],
            "year": r["year"],
            "journal": r["journal"],
            "source_ncd_title": r.get("source_ncd_title", ""),
            "source_ncd_number": r.get("source_ncd_number", ""),
        })
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_and_save()
