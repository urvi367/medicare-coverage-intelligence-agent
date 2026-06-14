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
# Uses flash (not flash-lite): flash-lite cannot reliably do the name-collision
# disambiguation (it labels same-name-different-thing abstracts as a Coverage Gap),
# whereas flash correctly resolves them to Insufficient Evidence. Independence comes
# from a different prompt + raw evidence (and human adjudication on top), not the tier.
LABEL_MODEL = "gemini-2.5-flash"

# Action-oriented alignment rubric — each label maps to one analyst action. Organized
# by DIRECTION of divergence (which side is ahead), not CMS's rhetorical stance, so the
# label tells the analyst what to do. MUST match the labels the pipeline emits
# (_GAP_SYSTEM). canonical_alignment matches longest-first, so "Partial Coverage Gap" is
# never swallowed by the "Coverage Gap" substring.
_ALIGNMENT_LABELS = [
    "Aligned",                # CMS & evidence agree (both cover/support or both decline) — no action
    "Partial Coverage Gap",   # CMS covers but narrower than evidence supports — broaden criteria
    "Coverage Gap",           # evidence supports but CMS does not cover/denies — expand · appeal
    "Overcoverage",           # CMS covers but evidence is weak/absent/negative — utilization review
    "Insufficient Evidence",  # retrieved abstracts can't support a judgment — manual review
]

_QUESTION_PROMPT = """\
Write ONE short, natural question a provider-side appeals specialist would ask to \
check whether the clinical evidence supports Medicare's coverage of the service below.

Rules:
1. LEAD with the specific intervention, using the actual clinical terms from the title — \
this is the most important word. You may name the condition it is used for IF that is \
unambiguous from the title (e.g. "Alpha-fetoprotein" -> hepatocellular carcinoma; "Cochlear \
Implantation" -> hearing loss). But if the intervention has SEVERAL possible uses or you \
are unsure which one the NCD covers (e.g. "Biofeedback Therapy", "Cellular Therapy", "Laser \
Procedures"), ask about the intervention in GENERAL — do NOT guess a condition.
2. Keep it to ONE sentence, ~12-20 words, concrete and topic-forward.
3. You may include at most ONE evidence word ("evidence", "studies", or "trials"). Do NOT \
pad it with phrases like "randomized controlled trials, observational studies, improved \
health outcomes, Medicare beneficiaries" — that vocabulary buries the clinical topic.
4. Ask about the evidence FOR the service, not about coverage criteria.

Examples:
- Title "Acupuncture for Fibromyalgia" -> "What does the evidence show about acupuncture for fibromyalgia?"
- Title "Transcatheter Aortic Valve Replacement (TAVR)" -> "Is there evidence supporting TAVR for severe aortic stenosis?"
- Title "Biofeedback Therapy" (no condition) -> "What does the evidence show about biofeedback therapy?"

NCD title: {title}

Respond with ONLY the question, no explanation."""

_LABEL_PROMPT = """\
You are a senior Medicare evidence reviewer. Your job is to help a coverage analyst \
see, for ONE topic, where CMS coverage and the published clinical evidence diverge AND \
WHICH WAY — so the analyst knows what to act on. The direction of the divergence is the \
whole point: it determines the action.

Read the CMS policy text and the PubMed abstracts below, then choose EXACTLY one label:
- Aligned — CMS and the evidence agree: CMS covers it and the evidence supports it, OR \
CMS does not cover it and the evidence does not support it either. No action needed.
- Partial Coverage Gap — CMS covers it, but the evidence clearly supports a SUBSTANTIVE, \
clinically-distinct broader use the policy excludes: a different population, a separate \
indication, or a materially looser threshold. The expansion must be REAL, not a marginal \
restatement of the covered use (e.g. "recurrence detection" when CMS already covers \
"monitoring response to therapy" is the SAME use → Aligned, not Partial). When genuinely \
torn between Aligned and Partial, choose Aligned. Action: broaden criteria.
- Coverage Gap — the evidence supports the service but CMS does not cover it, or \
explicitly denies it. Evidence is ahead of policy. Action: expand coverage / appeal. \
THIS is the key actionable finding.
- Overcoverage — CMS covers it, but the on-topic evidence shows it does NOT work (clearly \
weak or negative results). Coverage is ahead of evidence. Action: utilization review. \
(Note: evidence merely *absent* is Insufficient Evidence, not Overcoverage.)
- Insufficient Evidence — fewer than 2 abstracts actually report clinical outcomes for \
THIS intervention applied to THIS condition. Abstracts about a different indication or \
population, methodology only, or background/history do NOT count, even if real studies.

Decision procedure — follow IN ORDER (do not jump to a gap label):
0. NAME-COLLISION CHECK: the abstracts were retrieved by the NCD's TOPIC NAME, so some may \
be about a DIFFERENT intervention that merely shares the name or a keyword — e.g. a modern \
therapy with the same name as the obsolete one the NCD describes ("cellular therapy" = \
lamb-cell injection in the policy, but CAR-T in the abstracts), or a different device for \
the same organ ("bladder stimulator" implant vs sacral neuromodulation). Read the CMS \
POLICY text to learn what the intervention ACTUALLY is, then treat an abstract as on-topic \
ONLY if it studies that SAME intervention for the SAME condition — not just a shared word. \
Silently discard name-collision abstracts.
1. EVIDENCE GATE (unconditional — apply it even when CMS covers the service): count the \
remaining on-topic abstracts that directly report clinical outcomes of THIS intervention \
for THIS condition. If fewer than 2 → Insufficient Evidence, STOP. A topic with no on-topic \
outcome evidence is Insufficient Evidence even if CMS clearly covers it — it is NOT Aligned \
(you cannot confirm agreement with no evidence) and NOT a gap. Do not rescue off-topic or \
name-collision abstracts with a scope argument.
2. Establish CMS's coverage position from the policy text. First ask: does CMS cover this \
service for ANY indication at all? — yes (fully), yes (only a narrow population/indication), \
or no (non-covered / explicitly denied for all indications)?
3. Establish what the on-topic evidence shows: supports the service, shows it does not \
work, or mixed.
4. Map CMS position against the evidence (by direction, not CMS's rhetoric):
   - CMS covers it (any indication) AND the evidence supports it → Aligned
   - CMS does NOT cover it for any indication AND the evidence does not support it → Aligned
   - CMS covers it but more narrowly than the evidence supports → Partial Coverage Gap
   - CMS does NOT cover it for ANY indication BUT the evidence supports it → Coverage Gap
   - CMS covers it BUT the on-topic evidence shows it does not work → Overcoverage
ALIGNED vs PARTIAL tie-breaker: Partial requires the evidence to support a SUBSTANTIVE, \
clinically-distinct indication/population/threshold CMS excludes. If the "broader" evidence \
is essentially the covered use restated, or the expansion is marginal or ambiguous, label \
Aligned. Default to Aligned when unsure — do not award Partial for a minor extension.
CRITICAL: if CMS covers the service for even one indication, it can ONLY be Aligned, \
Partial Coverage Gap, or Overcoverage — NEVER a full Coverage Gap. Full Coverage Gap is \
reserved for services CMS covers for NO indication. Whether CMS "passively does not cover" \
or "actively denies citing weak evidence" does not change the label — only direction does. \
There is no separate "conflicting" label.

Calibration examples (illustrative only — cite ONLY PMIDs from the abstracts below):

Example 1 — CMS covers acupuncture for chronic low-back pain; multiple RCTs and a \
systematic review in the abstracts show it improves chronic low-back pain.
ALIGNMENT: Aligned
RATIONALE: CMS covers it and the evidence supports it for the same condition — they agree.

Example 2 — CMS covers home oxygen only for PO2 <= 55 mmHg; abstracts show benefit in \
severe hypoxemia AND moderate hypoxemia (56-65 mmHg).
ALIGNMENT: Partial Coverage Gap
RATIONALE: CMS covers it but the evidence supports a broader eligible population than \
the threshold permits.

Example 3 — CMS explicitly lists thermography as non-covered; abstracts are two RCTs \
showing diagnostic accuracy comparable to standard imaging.
ALIGNMENT: Coverage Gap
RATIONALE: Evidence supports an intervention CMS does not cover at all.

Example 4 — CMS denies coverage of sublingual antigen therapy, stating it is "not \
proven safe and effective"; multiple RCTs in the abstracts show efficacy.
ALIGNMENT: Coverage Gap
RATIONALE: Evidence supports a service CMS denies — the action is the same as any \
non-coverage gap, regardless of CMS's stated rationale.

Example 5 — CMS covers vertebroplasty for osteoporotic fractures; the retrieved \
abstracts are two sham-controlled RCTs finding no benefit over placebo.
ALIGNMENT: Overcoverage
RATIONALE: CMS covers it while the evidence shows it does not work — coverage is ahead \
of the evidence.

Example 6 — CMS covers acupuncture for chronic low-back pain; the retrieved abstracts \
study acupuncture for post-stroke fatigue and fibromyalgia, not low-back pain.
ALIGNMENT: Insufficient Evidence
RATIONALE: The abstracts are about different conditions, so they do not bear on this \
coverage decision.

Example 7 — CMS covers a clotting-factor drug only for patients with major active \
bleeding who failed other therapies; the abstracts (RCTs) support its use ALSO for \
routine prophylaxis to prevent bleeds.
ALIGNMENT: Partial Coverage Gap
RATIONALE: CMS already covers the drug for one indication, so this is a narrowing, not a \
full non-coverage gap — the evidence supports broadening to prophylaxis.

Example 8 — CMS covers an epidural blood graft for post-spinal-tap headache; the \
retrieved abstracts do not report clinical outcomes for this procedure/condition.
ALIGNMENT: Insufficient Evidence
RATIONALE: There is no on-topic outcome evidence, so agreement cannot be confirmed — \
covered-with-no-evidence is Insufficient Evidence, not Aligned.

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


def regenerate_questions() -> None:
    """Regenerate ONLY the questions (topic-forward) for existing golden_gap.json
    records, preserving the adjudicated reference labels.

    Safe because the labeler scores from NCD text + abstracts and never reads the
    question — so the topic-forward question only changes what the *pipeline* retrieves
    at eval time, not the reference label. Rebuilds gap_questions.json. Re-run judge_gap
    with a fresh answer cache afterward (delete logs/gap_answers_cache_*.json).
    """
    if not GOLDEN_GAP_PATH.exists():
        raise FileNotFoundError(f"{GOLDEN_GAP_PATH} not found — run generate() first.")
    recs = json.loads(GOLDEN_GAP_PATH.read_text(encoding="utf-8"))
    questions: dict[str, str] = {}
    for i, r in enumerate(recs, 1):
        try:
            q = _groq_question(r["topic"])
            time.sleep(_GROQ_DELAY)
        except Exception as e:
            logger.warning("Groq failed for %s: %s — keeping old question", r["expected_ncd"], e)
            questions[r["expected_ncd"]] = r["question"]
            continue
        r["question"] = q
        questions[r["expected_ncd"]] = q
        logger.info("  [%d/%d] %s -> %s", i, len(recs), r["expected_ncd"], q[:75])
        GOLDEN_GAP_PATH.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        QUESTIONS_PATH.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    logger.info("Regenerated %d questions (labels preserved)", len(recs))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "--regen-questions" in sys.argv:
        regenerate_questions()
    else:
        generate()
