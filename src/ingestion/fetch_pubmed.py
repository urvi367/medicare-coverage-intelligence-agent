"""Fetch PubMed abstracts for each NCD topic via NCBI E-utilities."""

import json
import logging
import re
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


# PubMed publication-type/MeSH filter that prioritises primary clinical evidence over narrative
# reviews. Meta-analyses and systematic reviews are syntheses but rank highest in the evidence
# tier, so they are included; the rest are primary designs. Cohort studies frequently lack a
# ptyp tag, so the MeSH term is included as well.
_PRIMARY_EVIDENCE_FILTER = (
    "(Meta-Analysis[ptyp] OR systematic[sb] OR Randomized Controlled Trial[ptyp] "
    "OR Controlled Clinical Trial[ptyp] OR Clinical Trial[ptyp] OR Observational Study[ptyp] "
    "OR Comparative Study[ptyp] OR Cohort Studies[Mesh])"
)


def _search_topic_pmids(title: str, n: int) -> list[str]:
    """PMIDs for a topic, prioritising primary evidence then backfilling to n.

    Pass 1 restricts to primary-evidence publication types/subsets; pass 2 backfills any
    shortfall with an unrestricted relevance search so rare or obsolete topics (where no
    primary evidence exists) still return abstracts rather than nothing.
    """
    primary = _search_pmids(f"({title}) AND {_PRIMARY_EVIDENCE_FILTER}", n)
    if len(primary) >= n:
        return primary[:n]
    time.sleep(_DELAY)
    seen = set(primary)
    backfill = [p for p in _search_pmids(title, n + len(primary)) if p not in seen]
    return (primary + backfill)[:n]


# Canonical PubMed "humans" filter: drops animal-only studies but keeps human AND
# not-yet-MeSH-indexed records. Broad LCD titles ("Special Histochemical Stains") otherwise
# collide with veterinary / food-science work; NCD titles are specific so they don't need it.
_HUMAN_FILTER = "NOT (animals[Mesh:noexp] NOT humans[Mesh:noexp])"


def _lcd_core_term(title: str) -> str:
    """The lead intervention phrase of a long LCD title (text before a ' for/in/of/with…'
    clause). Long compound LCD titles otherwise pull PubMed relevance toward the trailing
    disease name (e.g. allogeneic HCT 'for relapsed lymphoma' returns lymphoma-drug trials).
    Short titles (<= 8 words) are kept whole — their disease term is the point.
    """
    if len(title.split()) <= 8:
        return title
    head = re.split(r"\s+\b(?:for|in|of|with|due to|associated with|caused by)\b\s+",
                    title, maxsplit=1, flags=re.I)[0]
    return head.strip() or title


def _search_lcd_pmids(title: str, n: int) -> list[str]:
    """LCD-scoped search: human studies only, anchored on the core intervention term for
    long titles, primary evidence first then a human-restricted relevance backfill.

    Fixes the broad-LCD-title failure modes (veterinary collisions, trailing-disease drift,
    review floods) that title-only search produces. NCD evidence keeps `_search_topic_pmids`
    so the existing NCD golden set is unaffected.
    """
    # ANDed core words (not an exact phrase — "allogeneic stem cell transplant" should still
    # match) keep relevance on the intervention while excluding the trailing-disease drift.
    anchor = f"({_lcd_core_term(title)})"
    primary = _search_pmids(f"{anchor} AND {_PRIMARY_EVIDENCE_FILTER} {_HUMAN_FILTER}", n)
    if len(primary) >= n:
        return primary[:n]
    time.sleep(_DELAY)
    seen = set(primary)
    backfill = [p for p in _search_pmids(f"{anchor} {_HUMAN_FILTER}", n + len(primary))
                if p not in seen]
    return (primary + backfill)[:n]


def _extract_year(article: ElementTree.Element) -> str:
    """Extract a 4-digit publication year from a PubmedArticle element.

    NOTE: ElementTree treats a childless Element (like <Year>2020</Year>) as falsy,
    so `find(a) or find(b)` silently skips a real <Year>. Use explicit None checks.
    """
    for path in (".//Article//PubDate/Year", ".//PubDate/Year", ".//ArticleDate/Year"):
        el = article.find(path)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()[:4]
    # MedlineDate is free-form, e.g. "2020 Jan-Feb" or "Spring 2021" — pull first year.
    md = article.find(".//PubDate/MedlineDate")
    if md is not None and md.text:
        m = re.search(r"\b(\d{4})\b", md.text)
        if m:
            return m.group(1)
    return ""


# PubMed PublicationType values, strongest study design first. NLM tags every record
# with these; we keep the highest-ranked one as the study type for Evidence Grade.
_EVIDENCE_HIERARCHY = [
    "Meta-Analysis",
    "Systematic Review",
    "Randomized Controlled Trial",
    "Controlled Clinical Trial",
    "Clinical Trial",
    "Multicenter Study",
    "Comparative Study",
    "Observational Study",
    "Case Reports",
    "Practice Guideline",
    "Guideline",
    "Review",
]
# Funding/format/editorial PublicationTypes that are NOT study designs — never use these as
# the study_type, even as a fallback (otherwise the alphabetical fallback below mislabels e.g.
# "Research Support, N.I.H., Extramural" as a design).
_GENERIC_PUB_TYPES = {"Journal Article", "English Abstract", "Published Erratum",
                      "Research Support, Non-U.S. Gov't", "Research Support, U.S. Gov't, Non-P.H.S.",
                      "Research Support, U.S. Gov't, P.H.S.", "Research Support, N.I.H., Extramural",
                      "Research Support, N.I.H., Intramural",
                      "Research Support, American Recovery and Reinvestment Act",
                      "Comment", "Editorial", "Letter", "News", "Historical Article", "Biography",
                      "Portrait", "Autobiography", "Address", "Congress", "Lecture", "Overall"}


def _extract_study_type(article: ElementTree.Element) -> str:
    """Return the strongest study design from an article's PublicationType tags.

    Falls back to any non-generic type, else "" (e.g. a plain Journal Article).
    """
    types = {el.text for el in article.findall(".//PublicationType") if el.text}
    for t in _EVIDENCE_HIERARCHY:
        if t in types:
            return t
    meaningful = types - _GENERIC_PUB_TYPES
    if meaningful:
        return sorted(meaningful)[0]
    # Fallback: PubMed often tags an observational design via MeSH rather than PublicationType.
    # Infer a cohort/observational design from study-design MeSH headings so these don't fall
    # to "unspecified" (they are genuine moderate-tier primary evidence).
    mesh = {el.text for el in article.findall(".//MeshHeading/DescriptorName") if el.text}
    if mesh & {"Cohort Studies", "Case-Control Studies", "Cross-Sectional Studies",
               "Prospective Studies", "Retrospective Studies", "Longitudinal Studies",
               "Follow-Up Studies"}:
        return "Observational Study"
    return ""


# Evidence tier for gap-analysis grading, mapping NLM PublicationType -> a single ranked tier.
# Tiers (strongest first): T1 meta-analysis/systematic review; T2 RCT; T3 clinical trial;
# T4 cohort/observational/comparative; T5 case report/series; background = review/guideline
# (synthesis, not primary evidence); unspecified = no design tag (treat as background/weak).
def evidence_tier(study_type: str) -> str:
    """Map a PublicationType string to a gap-analysis evidence tier label."""
    t = (study_type or "").strip()
    if t in ("Meta-Analysis", "Systematic Review"):
        return "T1 highest (meta-analysis/systematic review)"
    if t == "Randomized Controlled Trial":
        return "T2 strong (RCT)"
    if t.startswith("Clinical Trial") or t == "Controlled Clinical Trial":
        return "T3 moderate-strong (clinical trial)"
    if t in ("Observational Study", "Comparative Study", "Multicenter Study",
             "Evaluation Study", "Validation Study", "Cohort Studies"):
        return "T4 moderate (cohort/observational)"
    if t in ("Case Reports", "Case Report"):
        return "T5 weak (case report/series)"
    if t in ("Review", "Scoping Review", "Practice Guideline", "Guideline",
             "Consensus Development Conference"):
        return "background only (review/guideline, not primary evidence)"
    return "unspecified (treat as background/weak)"


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

        year = _extract_year(article)

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        if abstract:
            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "study_type": _extract_study_type(article),
            })
    return records


def fetch_and_save(max_per_topic: int = 12) -> Path:
    """Search PubMed for each NCD title, fetch abstracts, and save to data/pubmed_raw.json.

    Uses a primary-evidence-first search per topic (see _search_topic_pmids) so the corpus is
    biased toward RCTs/meta-analyses/cohort studies rather than narrative reviews.

    Args:
        max_per_topic: Max abstracts to fetch per NCD topic (default 12, matches the gap
            analysis evidence budget in the labeler and pipeline).

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
        pmids = _search_topic_pmids(topic["title"], max_per_topic)
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


def _fetch_fields(pmids: list[str]) -> dict[str, dict[str, str]]:
    """Efetch a batch of PMIDs and return {pmid: {year, study_type}} for each."""
    if not pmids:
        return {}
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract"}
    try:
        r = _session.get(EFETCH_URL, params=params, timeout=60)
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)
    except Exception as e:
        logger.warning("efetch (fields) failed for %d pmids: %s", len(pmids), e)
        return {}
    out: dict[str, dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        if pmid:
            out[pmid] = {"year": _extract_year(article), "study_type": _extract_study_type(article)}
    return out


def backfill_metadata(batch_size: int = 200) -> Path:
    """Backfill missing year + study_type onto existing pubmed_raw.json (no re-search).

    Re-fetches only metadata for PMIDs already saved, preserving the exact dataset.
    Run `python -m src.rag.pubmed_indexer` afterward to rebuild the index.
    """
    path = DATA_DIR / "pubmed_raw.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    pmids = [r["pmid"] for r in records if r.get("pmid")]
    logger.info("Backfilling year + study_type for %d PMIDs in batches of %d...", len(pmids), batch_size)

    field_map: dict[str, dict[str, str]] = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        time.sleep(_DELAY)
        field_map.update(_fetch_fields(batch))
        logger.info("  [%d/%d] resolved %d records so far", min(i + batch_size, len(pmids)), len(pmids), len(field_map))

    filled_year = filled_type = 0
    for r in records:
        m = field_map.get(r.get("pmid", ""), {})
        if m.get("year") and not r.get("year"):
            r["year"] = m["year"]
            filled_year += 1
        if m.get("study_type") and not r.get("study_type"):
            r["study_type"] = m["study_type"]
            filled_type += 1

    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    typed = sum(1 for r in records if r.get("study_type"))
    logger.info("Filled %d years, %d study_types. %d/%d records now have a study_type. Saved %s",
                filled_year, filled_type, typed, len(records), path)
    return path


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
            "study_type": r.get("study_type", ""),
            "source_ncd_title": r.get("source_ncd_title", ""),
            "source_ncd_number": r.get("source_ncd_number", ""),
            "source_lcd_title": r.get("source_lcd_title", ""),
            "source_lcd_number": r.get("source_lcd_number", ""),
        })
    return docs


def fetch_lcd_evidence(max_per_topic: int = 8) -> Path:
    """Fetch PubMed evidence per active LCD topic (by title) and APPEND to
    pubmed_raw.json, tagged with source_lcd_number — so LCD gap analysis has topical
    evidence the same way NCDs do (filtered topical join, not open search). Idempotent:
    drops any prior LCD-sourced records first.

    Note: LCD titles are broader than NCD intervention names (e.g. "Plastic Surgery"),
    so the evidence is correspondingly broader than the NCD corpus.
    """
    lcds = json.loads((DATA_DIR / "lcd_raw.json").read_text(encoding="utf-8"))
    topics: dict[str, str] = {}
    for r in lcds:
        did = r.get("document_display_id")
        if did and r.get("title") and (r.get("retirement_date") or "N/A").strip() == "N/A":
            topics[did] = r["title"]
    logger.info("Fetching PubMed for %d LCD topics (max %d each)...", len(topics), max_per_topic)

    path = DATA_DIR / "pubmed_raw.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records = [r for r in records if not r.get("source_lcd_number")]  # idempotent re-run

    for i, (lcd_id, title) in enumerate(topics.items(), 1):
        if i % 50 == 0:
            logger.info("  ...%d/%d LCD topics", i, len(topics))
        for rec in _fetch_abstracts(_search_lcd_pmids(title, max_per_topic)):
            records.append({**rec, "source_lcd_title": title, "source_lcd_number": lcd_id})

    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved %d total PubMed records (NCD + LCD) to %s", len(records), path)
    return path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "--backfill" in sys.argv:
        backfill_metadata()  # patch year + study_type onto existing data, no re-search
    elif "--lcd" in sys.argv:
        fetch_lcd_evidence()  # append LCD-topic evidence to the corpus
    else:
        fetch_and_save()
