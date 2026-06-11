"""Generate a golden dataset for gap-analysis evaluation.

For each NCD in ncd_raw.json, generates one evidence-seeking question via Groq,
then runs gap_analysis() once to obtain reference labels (alignment, NCD cited,
PMIDs cited). Saved to data/golden_gap.json. Supports resuming.
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

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"

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


def _groq_question(title: str) -> str:
    """Generate an evidence-seeking question for an NCD topic via Groq."""
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
    r = requests.post(_GROQ_URL, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _parse_alignment(text: str) -> str:
    """Extract the Alignment label from a structured gap report."""
    m = re.search(r"Alignment:\s*\**(.+?)\**(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def _parse_pmids(text: str) -> list[str]:
    """Extract PMIDs cited in a gap report."""
    return list(set(re.findall(r"PMID\s*(\d+)", text, re.IGNORECASE)))


def generate(max_ncds: int | None = None) -> list[dict]:
    """Generate gap golden dataset from NCD topics.

    Args:
        max_ncds: Cap on number of NCDs to process (None = all).

    Returns:
        List of golden gap records, each with question, topic, expected_ncd,
        expected_alignment, reference_pmids, and reference_report.
    """
    from src.rag.pipeline import gap_analysis

    ncd_records = json.loads(NCD_PATH.read_text(encoding="utf-8"))
    ncds = [r for r in ncd_records if r.get("title") and r.get("text")]
    if max_ncds:
        ncds = ncds[:max_ncds]

    existing: list[dict] = []
    if GOLDEN_GAP_PATH.exists():
        existing = json.loads(GOLDEN_GAP_PATH.read_text(encoding="utf-8"))
        logger.info("Resuming — %d records already generated", len(existing))

    done_ncds = {r["expected_ncd"] for r in existing}

    for i, ncd in enumerate(ncds, 1):
        policy_number = ncd.get("document_display_id", "")
        if policy_number in done_ncds:
            logger.info("  Skipping [%d/%d] (done): %s", i, len(ncds), ncd["title"][:60])
            continue

        logger.info("  Generating [%d/%d]: %s", i, len(ncds), ncd["title"][:60])

        try:
            question = _groq_question(ncd["title"])
        except Exception as e:
            logger.warning("Groq failed for %s: %s — skipping", ncd["title"][:40], e)
            continue

        time.sleep(0.4)  # Groq free tier: ~30 RPM

        try:
            result = gap_analysis(question)
        except Exception as e:
            logger.warning("gap_analysis failed: %s — skipping", e)
            continue

        time.sleep(1.0)  # Gemini rate-limit headroom

        record = {
            "question": question,
            "topic": ncd["title"],
            "expected_ncd": policy_number,
            "expected_alignment": _parse_alignment(result["gap_report"]),
            "reference_pmids": _parse_pmids(result["gap_report"]),
            "reference_report": result["gap_report"],
        }
        existing.append(record)
        done_ncds.add(policy_number)
        GOLDEN_GAP_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info(
            "    Alignment: %s | PMIDs: %s",
            record["expected_alignment"],
            record["reference_pmids"],
        )

    logger.info("Done — %d gap golden records at %s", len(existing), GOLDEN_GAP_PATH)
    return existing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate()
