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
# Questions cache (keyed by NCD number) — decoupled from labeling so an interrupted
# run never re-calls Groq for a question it already generated (protects free-tier quota).
QUESTIONS_PATH = Path(__file__).parents[2] / "data" / "gap_questions.json"

_GROQ_DELAY = 2.0  # free tier ~30 RPM → 2s gap keeps well under the limit

# Truncation caps for the labeler context. Set generously — flash-lite has a huge
# context window, so the cost of full text is negligible and avoids cutting the
# coverage decision (NCD: indications_limitations is appended last) or an abstract's
# conclusion (p90 abstract ~2465 chars; the old 600 cap cut 97% of abstracts).
_MAX_NCD_CHARS = 8000
_MAX_ABSTRACT_CHARS = 2500

# NCDs whose coverage is restricted by statute/law rather than clinical evidence.
# The evidence-vs-coverage gap framing does not apply (no amount of evidence changes a
# legal restriction), so they are excluded from the gap golden set. Extend as found.
# NOTE: the 210.x screening series ("statutory" preventive services) are NOT here — those
# are statutorily *mandated*, evidence-based coverage, valid for gap analysis.
_STATUTORY_EXCLUSIONS = {
    "140.1",  # Abortion — coverage restricted by the Hyde Amendment, not evidence
    "140.4",  # Plastic Surgery to Correct "Moon Face" — cosmetic exclusion, §1862(a)(10)
}

# Retired / rescinded / superseded NCDs have no current coverage position, so the
# evidence-vs-coverage gap framing is meaningless. Detected by the title marker CMS
# stamps on them (e.g. "- RETIRED", "(Replaced with Section 220.6.17)").
_RETIRED_RE = re.compile(r"-\s*RETIRED|\bRETIRED\b|rescinded|replaced with section", re.I)

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
- Partially Aligned — the abstracts directly study the SAME intervention for the SAME \
condition CMS addresses, agree it works, but differ on scope (narrower/broader \
population, stricter thresholds, severity). Scope mismatch ONLY — not evidence about \
a different condition or indication.
- Conflicting — opposite directions: CMS covers what the evidence says does not work, \
or CMS's stated rationale contradicts the evidence
- Coverage Gap — evidence supports the intervention but CMS explicitly does not cover \
it (a non-coverage position, not mere narrowness)
- Inverse Gap — CMS covers it but the retrieved evidence is weak, absent, or negative
- Insufficient Evidence — fewer than 2 abstracts actually report clinical outcomes for \
THIS intervention applied to THIS condition. Abstracts about a different indication, a \
different population, methodology only, or background/history do NOT count as relevant \
evidence even if they are real studies.

FIRST, before any other label, apply this gate:
  Count the abstracts that directly report clinical outcomes of THIS intervention for \
THIS condition. If fewer than 2 → ALIGNMENT: Insufficient Evidence. Do NOT rescue \
off-topic abstracts by inventing a scope or "fewer indications" argument — evidence for \
a different condition is not evidence that CMS coverage is too narrow.

Decision rules for the confusable boundaries:
- Partially Aligned vs Insufficient Evidence: Partially Aligned requires on-topic \
evidence about the same intervention+condition. If the abstracts are about a different \
indication/population, that is Insufficient Evidence, not Partially Aligned.
- Partially Aligned vs Conflicting: if CMS and the evidence agree the intervention \
works but disagree on WHO/WHEN it is appropriate, that is Partially Aligned. \
Conflicting requires opposite conclusions about whether it works at all.
- Coverage Gap vs Conflicting: Coverage Gap requires an explicit CMS non-coverage \
position on something the evidence supports. If CMS covers it (even restrictively), \
it is never a Coverage Gap.

Calibration examples (illustrative only — cite ONLY PMIDs from the abstracts below):

Example 1 — CMS covers home oxygen for patients with PO2 <= 55 mmHg; abstracts show \
benefit in severe hypoxemia AND moderate hypoxemia (56-65 mmHg).
ALIGNMENT: Partially Aligned
RATIONALE: Both agree oxygen therapy works, but the evidence supports a broader \
eligible population than CMS's threshold permits — a scope mismatch, not opposite \
conclusions.

Example 2 — CMS explicitly lists thermography as non-covered; abstracts are two RCTs \
showing diagnostic accuracy comparable to standard imaging.
ALIGNMENT: Coverage Gap
RATIONALE: Explicit non-coverage of an intervention the retrieved evidence supports.

Example 3 — CMS covers vertebroplasty for osteoporotic fractures; the retrieved \
abstracts are two sham-controlled RCTs finding no benefit over placebo.
ALIGNMENT: Conflicting
RATIONALE: CMS covers it while the evidence concludes it does not work — opposite \
directions on efficacy, not a scope question.

Example 4 — CMS covers a device for heart-failure monitoring; the only retrieved \
abstracts are one case report and one engineering-methods paper with no clinical \
outcomes.
ALIGNMENT: Insufficient Evidence
RATIONALE: Fewer than 2 abstracts address clinical outcomes for this intervention, \
so alignment cannot be judged.

Example 5 — CMS covers acupuncture for chronic low-back pain; the retrieved abstracts \
are real RCTs, but they study acupuncture for post-stroke fatigue and fibromyalgia, \
not low-back pain.
ALIGNMENT: Insufficient Evidence
RATIONALE: The abstracts study the same intervention for DIFFERENT conditions, so they \
do not bear on CMS's low-back-pain coverage — this is not a "fewer indications" scope \
mismatch.

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
            "study_type": meta.get("study_type", ""),
            "text": text,
        })
    return out[:limit]


def _format_abstracts(abstracts: list[dict]) -> str:
    if not abstracts:
        return "No abstracts retrieved for this topic."
    parts = []
    for a in abstracts:
        stype = a.get("study_type") or "study type unspecified"
        header = f"PMID {a['pmid']} ({a.get('year') or '?'}, {stype}, {a.get('journal') or '?'})"
        parts.append(f"{header}\n{a['text'][:_MAX_ABSTRACT_CHARS]}")
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
        ncd_text=ncd_text[:_MAX_NCD_CHARS],
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
    from src.ingestion.fetch import load_documents
    from src.rag.pubmed_indexer import load_pubmed_index

    # load_documents assembles + HTML-cleans the decision-bearing NCD fields
    # (item_service_description + indications_limitations) into "text" — the same
    # content the index is built from. ncd_raw.json has no top-level "text" field.
    ncds = load_documents("ncd")
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
        policy_number = ncd["policy_number"]
        if policy_number in done_ncds:
            logger.info("  Skipping [%d/%d] (done): %s", i, len(ncds), ncd["title"][:60])
            continue
        if policy_number in _STATUTORY_EXCLUSIONS:
            logger.info("  Skipping [%d/%d] (statutory restriction, not evidence-based): %s",
                        i, len(ncds), ncd["title"][:60])
            continue
        if _RETIRED_RE.search(ncd["title"]):
            logger.info("  Skipping [%d/%d] (retired/superseded NCD): %s",
                        i, len(ncds), ncd["title"][:60])
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
