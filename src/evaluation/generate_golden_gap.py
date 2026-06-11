"""Generate a golden dataset for gap-analysis evaluation.

For each NCD, generates one evidence-seeking question via Groq, then labels the
reference alignment with an INDEPENDENT Gemini judge that reads the raw NCD policy
text + the abstracts for that NCD — NOT the pipeline's own gap report. This breaks
the circularity of grading the pipeline against its own output: the reference label
is produced by a separate prompt reasoning from raw evidence.

Saved to data/golden_gap.json. Supports resuming.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_GAP_PATH = Path(__file__).parents[2] / "data" / "golden_gap.json"
NCD_PATH = Path(__file__).parents[2] / "data" / "ncd_raw.json"
# Questions cache (keyed by NCD number) — decoupled from labeling so an interrupted
# run never re-calls Groq for a question it already generated (protects free-tier quota).
QUESTIONS_PATH = Path(__file__).parents[2] / "data" / "gap_questions.json"

_GROQ_DELAY = 2.0  # free tier ~30 RPM → 2s gap keeps well under the limit

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"

# Independent labeler. Reads raw NCD + abstracts; never sees the pipeline's report.
# flash-lite: cheaper, and a different tier from the pipeline's flash generator —
# marginally stronger independence for the reference labels.
LABEL_MODEL = "gemini-2.5-flash-lite"

# Alignment rubric — MUST match the labels the pipeline emits (_GAP_SYSTEM) so the
# pipeline's output can be compared against the reference.
_ALIGNMENT_LABELS = [
    "Aligned",
    "Partially Aligned",
    "Conflicting",
    "Coverage Gap",
    "Inverse Gap",
    "Insufficient Evidence",
]

_QUESTION_PROMPT = """\
You are generating evaluation data for a Medicare coverage intelligence system.

Given the NCD (National Coverage Determination) title below, write one question \
that:
1. Asks about the clinical evidence behind the coverage decision
2. Sounds like a clinician, researcher, or health policy analyst would ask
3. Uses words like "evidence", "studies", "clinical data", "RCTs", or "research"
4. Does NOT ask about coverage criteria — only about the supporting evidence

NCD title: {title}

Respond with ONLY the question, no explanation."""

_LABEL_PROMPT = """\
You are a senior Medicare evidence reviewer. Independently judge how well CMS \
coverage policy aligns with the published clinical evidence for ONE topic.

Read the CMS policy text and the PubMed abstracts below, then decide the alignment \
using EXACTLY one of these labels:
- Aligned — CMS coverage matches what the evidence supports
- Partially Aligned — broadly consistent but with notable caveats or mismatched scope
- Conflicting — CMS position and the evidence point in opposite directions
- Coverage Gap — evidence supports the intervention but CMS does not cover it
- Inverse Gap — CMS covers it but the clinical evidence is weak or absent
- Insufficient Evidence — too few/weak abstracts to judge alignment at all

Reason ONLY from the documents provided. Do not use outside knowledge of newer policy.

Respond in EXACTLY this format:
ALIGNMENT: <one label from the list above>
KEY_PMIDS: <comma-separated PMIDs most decisive for your judgment, or NONE>
RATIONALE: <one sentence>

CMS POLICY ({ncd_number} — {ncd_title}):
{ncd_text}

PUBMED ABSTRACTS:
{abstracts}"""


def _groq_question(title: str, retries: int = 6) -> str:
    """Generate an evidence-seeking question for an NCD topic via Groq.

    Retries on 429 using the Retry-After header (Groq free tier rate limit).
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": _QUESTION_PROMPT.format(title=title)}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    for attempt in range(retries):
        r = requests.post(_GROQ_URL, headers=headers, json=body, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", min(2 ** attempt, 60))) + 1
            logger.warning("Groq rate limited (attempt %d/%d) — waiting %.0fs", attempt + 1, retries, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"Groq still rate-limited after {retries} retries (daily quota may be exhausted).")


def _load_questions_cache() -> dict[str, str]:
    if QUESTIONS_PATH.exists():
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {}


def _abstracts_for_ncd(pubmed_db, ncd_number: str, limit: int = 12) -> list[dict]:
    """Return up to `limit` abstract dicts (pmid, year, journal, text) for an NCD."""
    if not ncd_number:
        return []
    got = pubmed_db.get(where={"source_ncd_number": ncd_number}, include=["documents", "metadatas"])
    out = []
    for text, meta in zip(got["documents"], got["metadatas"]):
        out.append({
            "pmid": meta.get("pmid", ""),
            "year": meta.get("year", ""),
            "journal": meta.get("journal", ""),
            "text": text,
        })
    return out[:limit]


def _format_abstracts(abstracts: list[dict]) -> str:
    if not abstracts:
        return "No abstracts retrieved for this topic."
    parts = []
    for a in abstracts:
        header = f"PMID {a['pmid']} ({a.get('year') or '?'}, {a.get('journal') or '?'})"
        parts.append(f"{header}\n{a['text'][:600]}")
    return "\n\n---\n\n".join(parts)


def canonical_alignment(text: str) -> str:
    """Map a free-form alignment string to a canonical rubric label, or "" if none.

    Matches longest label first so "Partially Aligned" is not captured by "Aligned"
    (a substring). Shared by the labeler and judge_gap so both sides compare on the
    same canonical vocabulary regardless of trailing em-dash descriptors or notes.
    """
    low = text.lower()
    for lab in sorted(_ALIGNMENT_LABELS, key=len, reverse=True):
        if lab.lower() in low:
            return lab
    return ""


def _parse_label(text: str) -> dict:
    """Parse the structured judge response into alignment, key_pmids, rationale."""
    align_m = re.search(r"ALIGNMENT:\s*(.+)", text)
    pmid_m = re.search(r"KEY_PMIDS:\s*(.+)", text)
    rat_m = re.search(r"RATIONALE:\s*(.+)", text)

    alignment = align_m.group(1).strip() if align_m else ""
    canon = canonical_alignment(alignment) or alignment

    pmids: list[str] = []
    if pmid_m and "none" not in pmid_m.group(1).lower():
        pmids = re.findall(r"\d{5,}", pmid_m.group(1))

    return {
        "alignment": canon,
        "key_pmids": pmids,
        "rationale": rat_m.group(1).strip() if rat_m else "",
    }


def _judge_alignment(ncd_number: str, ncd_title: str, ncd_text: str, abstracts: list[dict]) -> dict:
    """Independently label alignment from raw NCD text + abstracts via Gemini."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=LABEL_MODEL, temperature=0)
    prompt = _LABEL_PROMPT.format(
        ncd_number=ncd_number,
        ncd_title=ncd_title,
        ncd_text=ncd_text[:3000],
        abstracts=_format_abstracts(abstracts),
    )
    for attempt in range(8):
        try:
            return _parse_label(llm.invoke(prompt).content)
        except Exception as exc:
            s = str(exc)
            # A monthly spending cap won't clear on retry — fail fast so generation
            # stops cleanly instead of backing off for minutes per call.
            if "spend" in s.lower() or "spending cap" in s.lower():
                raise RuntimeError(
                    "Gemini monthly spending cap exceeded — raise the cap at "
                    "https://ai.studio/spend, then re-run (golden gen resumes)."
                ) from exc
            m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", s)
            retryable = bool(m) or any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "503", "SERVICE_UNAVAILABLE"))
            if not retryable or attempt == 7:
                raise
            wait = float(m.group(1)) + 2 if m else min(2 ** attempt * 5, 60)
            logger.warning("Labeler rate limited — waiting %.0fs", wait)
            time.sleep(wait)


def generate(max_ncds: int | None = None) -> list[dict]:
    """Generate gap golden dataset with independently-labeled reference alignment.

    Args:
        max_ncds: Cap on number of NCDs to process (None = all).

    Returns:
        List of golden gap records.
    """
    from src.rag.pubmed_indexer import load_pubmed_index

    ncd_records = json.loads(NCD_PATH.read_text(encoding="utf-8"))
    ncds = [r for r in ncd_records if r.get("title") and r.get("text")]
    if max_ncds:
        ncds = ncds[:max_ncds]

    pubmed_db = load_pubmed_index()

    existing: list[dict] = []
    if GOLDEN_GAP_PATH.exists():
        existing = json.loads(GOLDEN_GAP_PATH.read_text(encoding="utf-8"))
        logger.info("Resuming — %d records already generated", len(existing))

    questions = _load_questions_cache()
    if questions:
        logger.info("Loaded %d cached questions", len(questions))

    done_ncds = {r["expected_ncd"] for r in existing}

    for i, ncd in enumerate(ncds, 1):
        policy_number = ncd.get("document_display_id", "")
        if policy_number in done_ncds:
            logger.info("  Skipping [%d/%d] (done): %s", i, len(ncds), ncd["title"][:60])
            continue

        logger.info("  Labeling [%d/%d]: %s", i, len(ncds), ncd["title"][:60])

        abstracts = _abstracts_for_ncd(pubmed_db, policy_number)
        if not abstracts:
            logger.info("    No abstracts for %s — skipping (no evidence to judge)", policy_number)
            continue

        if policy_number in questions:
            question = questions[policy_number]
        else:
            try:
                question = _groq_question(ncd["title"])
            except Exception as e:
                logger.warning("Groq failed for %s: %s — skipping", ncd["title"][:40], e)
                continue
            questions[policy_number] = question
            QUESTIONS_PATH.write_text(json.dumps(questions, indent=2), encoding="utf-8")
            time.sleep(_GROQ_DELAY)  # Groq free-tier rate-limit headroom

        try:
            label = _judge_alignment(policy_number, ncd["title"], ncd["text"], abstracts)
        except Exception as e:
            logger.warning("Labeler failed: %s — skipping", e)
            continue
        time.sleep(1.0)  # Gemini headroom

        record = {
            "question": question,
            "topic": ncd["title"],
            "expected_ncd": policy_number,
            "reference_alignment": label["alignment"],
            "reference_rationale": label["rationale"],
            "reference_pmids": label["key_pmids"],
            "n_abstracts_judged": len(abstracts),
        }
        existing.append(record)
        done_ncds.add(policy_number)
        GOLDEN_GAP_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("    → %s | key PMIDs: %s", record["reference_alignment"], record["reference_pmids"])

    logger.info("Done — %d gap golden records at %s", len(existing), GOLDEN_GAP_PATH)
    return existing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate()
