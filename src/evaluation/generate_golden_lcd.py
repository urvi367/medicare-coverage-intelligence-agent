"""Generate an LCD coverage-question eval set (Groq free tier).

For a stratified sample of indexed LCDs (across the 5 MACs), write a lay coverage
question and pair it with a state in that LCD's jurisdiction. `judge_lcd` then checks
the NCD→LCD cascade resolves the question — asked in that state — back to the source
LCD. Questions are deliberately lay (no "LCD"/"Medicare"/state mention) to test
body-based retrieval the way a real user would phrase it.

Output: data/golden_lcd.json. Usage: python -m src.evaluation.generate_golden_lcd
"""
import collections
import json
import logging
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
GOLDEN_LCD_PATH = DATA_DIR / "golden_lcd.json"

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"
_GROQ_DELAY = 2.0  # free tier ~30 RPM
_PER_MAC = 12      # sampled LCDs per MAC

# mac metadata flag -> (contractor name as in the data, a state that MAC serves)
_MAC_INFO = {
    "mac_noridian": ("Noridian Healthcare Solutions, LLC", "California"),
    "mac_cgs": ("CGS Administrators, LLC", "Ohio"),
    "mac_wps": ("WPS Insurance Corporation", "Indiana"),
    "mac_palmetto": ("Palmetto GBA", "Georgia"),
    "mac_ngs": ("National Government Services, Inc.", "New York"),
}

_Q_PROMPT = (
    "A Medicare Local Coverage Determination is titled: \"{title}\".\n"
    "Write ONE short, natural coverage question that a provider's billing or "
    "prior-auth staff would actually ask about whether this service is covered. "
    "Name the specific service/procedure in plain language. Do NOT mention 'LCD', "
    "'Medicare', 'coverage determination', or any US state. Output ONLY the question."
)


def _groq_question(title: str, retries: int = 6) -> str:
    """Generate one lay coverage question for an LCD title via Groq (free-tier backoff)."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": _Q_PROMPT.format(title=title)}],
        "temperature": 0.7,
        "max_tokens": 80,
    }
    for attempt in range(retries):
        r = requests.post(_GROQ_URL, headers=headers, json=body, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", min(2 ** attempt, 60))) + 1
            logger.warning("Groq rate limited (%d/%d) — waiting %.0fs", attempt + 1, retries, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip().strip('"')
    raise RuntimeError("Groq still rate-limited after retries (daily quota may be exhausted).")


def _lcds_by_mac() -> dict[str, list[tuple[str, str]]]:
    """One (lcd_id, title) per LCD, grouped by its (first) MAC flag, from the index."""
    from src.rag.pipeline import _get_db
    got = _get_db()._collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for m in got["metadatas"]:
        if m.get("source") != "LCD":
            continue
        pid = m.get("policy_number")
        if pid and pid not in seen:
            seen[pid] = m
    groups: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for pid, m in seen.items():
        title = (m.get("title") or "").strip()
        if not title:
            continue
        flag = next((k for k in _MAC_INFO if m.get(k)), None)
        if flag:
            groups[flag].append((pid, title))
    return groups


def main(seed: int = 42) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = random.Random(seed)
    groups = _lcds_by_mac()

    records = []
    for flag, (mac, state) in _MAC_INFO.items():
        pool = groups.get(flag, [])
        sample = rng.sample(pool, min(_PER_MAC, len(pool)))
        logger.info("%s: sampling %d of %d LCDs", mac, len(sample), len(pool))
        for pid, title in sample:
            try:
                q = _groq_question(title)
            except Exception as e:
                logger.warning("Groq failed for %s (%s) — skipping: %s", pid, title[:40], e)
                continue
            records.append({"question": q, "state": state, "expected_lcd": pid,
                            "mac": mac, "title": title})
            time.sleep(_GROQ_DELAY)

    GOLDEN_LCD_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d LCD eval records to %s", len(records), GOLDEN_LCD_PATH)


if __name__ == "__main__":
    main()
