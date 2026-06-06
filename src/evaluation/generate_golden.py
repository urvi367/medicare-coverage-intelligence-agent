"""Generate a golden Q&A dataset from ingested CMS documents using Groq."""

import json
import logging
import os
import random
import time
from pathlib import Path

from groq import Groq
from groq import RateLimitError
from dotenv import load_dotenv

from src.ingestion.fetch import load_documents

load_dotenv()
logger = logging.getLogger(__name__)

# llama-3.1-8b-instant free tier: 30 RPM, 6K TPM, 14.4K RPD, 500K TPD.
# Each request uses ~900 tokens → TPM is the binding constraint at ~6.5 req/min.
# 10s delay keeps us safely under 6K TPM. On a 429, the retry-after header
# gives the exact wait; 60s is the fallback (one full minute window reset).
_REQUEST_DELAY = 10.0
_RATE_LIMIT_WAIT = 60

GOLDEN_PATH = Path(__file__).parents[2] / "data" / "golden_dataset.json"

_PROMPT = """You are a Medicare coverage policy expert creating evaluation questions.

Given the policy document below, write ONE question that:
- Can be answered directly from the document
- Requires understanding coverage criteria (not just a yes/no)
- Would be asked by a healthcare provider or patient

Then write a concise reference answer (2-4 sentences) grounded only in this document.

Policy document:
{text}

Respond with valid JSON only, no markdown:
{{"question": "...", "reference_answer": "..."}}"""


def _generate_pair(client: Groq, doc: dict, max_retries: int = 5) -> dict | None:
    """Ask Groq to generate one Q&A pair from a single document."""
    text = doc["text"][:3000]

    for attempt in range(max_retries):
        try:
            msg = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            )
            raw = msg.choices[0].message.content.strip()
            pair = json.loads(raw)
            pair["policy_number"] = doc.get("policy_number", "")
            pair["title"] = doc.get("title", "")
            pair["source"] = doc.get("source", "")
            return pair
        except RateLimitError as e:
            # Prefer the exact retry-after Groq sends; fall back to escalating wait.
            retry_after = getattr(e.response, "headers", {}).get("retry-after")
            wait = int(float(retry_after)) + 1 if retry_after else _RATE_LIMIT_WAIT * (attempt + 1)
            logger.warning("Rate limited — waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Skipping doc %s — parse error: %s", doc.get("doc_id"), e)
            return None

    logger.warning("Skipping doc %s — exhausted retries", doc.get("doc_id"))
    return None


def _doc_key(doc: dict) -> str:
    """Stable identity key for a document, used to skip already-processed docs."""
    return f"{doc.get('source', '')}:{doc.get('policy_number') or doc.get('title', '')}"


def _save(pairs: list[dict]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(pairs, indent=2), encoding="utf-8")


def generate(
    n: int = 200,
    doc_types: tuple[str, ...] = ("ncd", "lcd"),
    seed: int = 42,
) -> list[dict]:
    """Sample n documents and generate one Q&A pair each, saving after every success.

    Resumes automatically from any existing golden_dataset.json, so re-running after
    hitting a daily token limit picks up where it left off.

    Args:
        n:         Target number of pairs.
        doc_types: Which document types to sample from.
        seed:      Random seed for reproducible sampling (must stay fixed across runs).

    Returns:
        List of dicts with keys: question, reference_answer, title, policy_number, source.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    all_docs: list[dict] = []
    for doc_type in doc_types:
        try:
            all_docs.extend(load_documents(doc_type))
        except FileNotFoundError:
            logger.warning("No %s data — run `python -m src.ingestion.fetch` first", doc_type.upper())

    if not all_docs:
        raise RuntimeError("No documents found. Run ingestion first.")

    # Load any pairs saved from a previous (possibly incomplete) run.
    pairs: list[dict] = []
    if GOLDEN_PATH.exists():
        pairs = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded %d existing pairs from %s", len(pairs), GOLDEN_PATH)

    done_keys: set[str] = {_doc_key(p) for p in pairs}

    # Same seed → same sample order every run, so skipping done_keys always resumes correctly.
    random.seed(seed)
    sample = random.sample(all_docs, min(n, len(all_docs)))
    remaining = [doc for doc in sample if _doc_key(doc) not in done_keys]

    if not remaining:
        logger.info("All %d pairs already generated — nothing to do.", len(pairs))
        return pairs

    logger.info(
        "%d/%d pairs already done. Generating %d more...",
        len(pairs), n, len(remaining),
    )

    for i, doc in enumerate(remaining, 1):
        logger.info("  [%d/%d] %s %s", i, len(remaining), doc.get("source", ""), doc.get("title", "")[:60])
        pair = _generate_pair(client, doc)
        if pair:
            pairs.append(pair)
            _save(pairs)  # persist immediately so progress survives a daily-limit cutoff
        time.sleep(_REQUEST_DELAY)

    logger.info("Total pairs saved: %d", len(pairs))
    return pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pairs = generate(n=200)
    print(f"\nDone — {len(pairs)} examples in {GOLDEN_PATH}")
