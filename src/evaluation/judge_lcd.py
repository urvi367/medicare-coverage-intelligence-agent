"""Evaluate the NCD→LCD cascade on the LCD coverage eval set (data/golden_lcd.json).

For each (question, state, expected_lcd), run resolve_governing_policy(question, state)
and check whether it resolves to the EXPECTED LCD. Because some LCD-topic services are
also NCD-governed, the cascade may legitimately return an NCD — reported separately, not
as an LCD error.

Metrics:
  disposition   — how questions resolved: lcd / ncd / none
  lcd_recall    — fraction resolved to the CORRECT LCD (over all records)
  lcd_precision — of those resolved to an LCD, fraction that were the right one
  ncd_intercept — fraction where an NCD governed instead (service has national coverage)

Usage: python -m src.evaluation.judge_lcd
"""
import collections
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
GOLDEN_LCD_PATH = Path(__file__).parents[2] / "data" / "golden_lcd.json"
# Generated LCD answers are cached here so re-scoring (e.g. disclaimer-stripped) never
# pays to regenerate them — generation is the avoidable API cost, RAGAS is the rest.
_FAITH_CACHE = Path(__file__).parents[2] / "data" / "lcd_faith_answers.json"


def evaluate(n_samples: int | None = None) -> dict[str, Any]:
    from src.rag.pipeline import resolve_governing_policy

    golden = json.loads(GOLDEN_LCD_PATH.read_text(encoding="utf-8"))
    if n_samples:
        golden = golden[:n_samples]
    n = len(golden)
    logger.info("Evaluating cascade on %d LCD records", n)

    disp = collections.Counter()
    lcd_hit = lcd_resolved = 0
    misses = []
    for i, r in enumerate(golden, 1):
        res = resolve_governing_policy(r["question"], state=r["state"])
        disp[res.source] += 1
        if res.source == "lcd":
            lcd_resolved += 1
            if r["expected_lcd"] in res.policy_ids:
                lcd_hit += 1
            else:
                misses.append((r["expected_lcd"], res.policy_ids[0] if res.policy_ids else "?", r["question"][:60]))
        else:
            misses.append((r["expected_lcd"], f"[{res.source}:{res.note}]", r["question"][:60]))
        if i % 10 == 0:
            logger.info("  ...%d/%d", i, n)

    scores = {
        "n": n,
        "disposition": dict(disp),
        "lcd_recall": round(lcd_hit / n, 3) if n else 0.0,
        "lcd_precision": round(lcd_hit / lcd_resolved, 3) if lcd_resolved else 0.0,
        "ncd_intercept": round(disp.get("ncd", 0) / n, 3) if n else 0.0,
        "none_rate": round(disp.get("none", 0) / n, 3) if n else 0.0,
    }
    return scores, misses


# The LCD answer prompt mandates an opening jurisdiction disclaimer ("based on the
# {mac} LCD ... coverage may differ in other regions") that is true but NOT in the LCD
# body — so RAGAS faithfulness flags it as ungrounded on every LCD answer. Stripping it
# isolates whether the low score is that artifact vs the model actually inventing criteria.
def _strip_disclaimer(answer_text: str) -> str:
    """Drop the mandated jurisdiction-disclaimer sentence(s) from an LCD answer."""
    keep = []
    for s in re.split(r"(?<=[.!?])\s+", answer_text):
        low = s.lower()
        is_disclaimer = (
            "may differ" in low
            or "jurisdiction only" in low
            or "other mac" in low
            or ("based on" in low and "lcd" in low and ("applies" in low or "jurisdiction" in low))
            or "this determination is based on an lcd" in low
        )
        if not is_disclaimer:
            keep.append(s)
    return " ".join(keep).strip() or answer_text


def _generate_lcd_answers(sample_size: int, n_faith: int, seed: int) -> list[dict]:
    """LCD-resolved (question, answer, context) triples — cached so we generate once."""
    if _FAITH_CACHE.exists():
        c = json.loads(_FAITH_CACHE.read_text(encoding="utf-8"))
        if c.get("seed") == seed and c.get("sample_size") == sample_size and c.get("n_faith") == n_faith:
            logger.info("loaded %d cached LCD answers — no regeneration", len(c["items"]))
            return c["items"]

    from src.rag.pipeline import answer
    golden = json.loads(GOLDEN_LCD_PATH.read_text(encoding="utf-8"))
    sample = random.Random(seed).sample(golden, min(sample_size, len(golden)))
    items = []
    for i, r in enumerate(sample, 1):
        res = answer(r["question"], state=r["state"])
        srcs = res.get("sources", [])
        if srcs and srcs[0].metadata.get("source") == "LCD":
            items.append({"q": r["question"], "a": res["answer"],
                          "ctx": [d.page_content for d in srcs]})
        logger.info("  generated %d/%d (LCD-resolved so far: %d)", i, len(sample), len(items))
    items = items[:n_faith]
    _FAITH_CACHE.write_text(json.dumps(
        {"seed": seed, "sample_size": sample_size, "n_faith": n_faith, "items": items},
        ensure_ascii=False), encoding="utf-8")
    logger.info("cached %d answers to %s", len(items), _FAITH_CACHE.name)
    return items


def evaluate_faithfulness(sample_size: int = 20, n_faith: int = 15, seed: int = 42,
                          strip_disclaimer: bool = False) -> dict[str, Any]:
    """RAGAS faithfulness of LCD-resolved answers vs the LCD body (reference-free).

    "Does the LCD answer invent coverage criteria not in the LCD body?" Answers are
    generated once and cached. With strip_disclaimer, the mandated jurisdiction
    disclaimer (true but not in the LCD body) is removed before scoring, to separate
    that metric artifact from real hallucination — compare against the un-stripped run.
    """
    from unittest.mock import MagicMock
    # ragas imports a Vertex chat model that isn't present in this langchain version
    sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

    import pandas as pd
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._faithfulness import Faithfulness

    items = _generate_lcd_answers(sample_size, n_faith, seed)
    if not items:
        return {"lcd_faithfulness": None, "n_faith": 0}

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0), bypass_n=True)

    def _score(question: str, response: str, ctx: list[str]) -> float:
        ds = EvaluationDataset(samples=[SingleTurnSample(
            user_input=question, response=response, retrieved_contexts=ctx)])
        for attempt in range(8):
            try:
                return float(ragas_evaluate(ds, metrics=[Faithfulness(llm=llm)]).to_pandas()["faithfulness"].iloc[0])
            except Exception as e:
                if attempt == 7 or not any(t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "503")):
                    raise
                time.sleep(min(2 ** attempt * 5 + 1, 90))

    scores = []
    for i, it in enumerate(items, 1):
        resp = _strip_disclaimer(it["a"]) if strip_disclaimer else it["a"]
        if i == 1 and strip_disclaimer:  # log one example so the strip is verifiable
            logger.info("  example full answer:\n%s", it["a"][:400])
            logger.info("  example stripped answer:\n%s", resp[:400])
        logger.info("  faithfulness %d/%d", i, len(items))
        scores.append(_score(it["q"], resp, it["ctx"]))

    key = "lcd_faithfulness_no_disclaimer" if strip_disclaimer else "lcd_faithfulness"
    return {key: round(float(pd.Series(scores).mean(skipna=True)), 3), "n_faith": len(items)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scores, misses = evaluate()
    print("\n=== LCD cascade eval ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print("\nmisses (expected -> got | question):")
    for exp, got, q in misses[:20]:
        print(f"  {exp:>8} -> {got:<22} {q}")

    if "--faithfulness" in sys.argv:
        # disclaimer-excluded: the mandated jurisdiction preamble is true but not in the
        # LCD body, so it's a RAGAS artifact (0.646 raw) — strip it for real grounding.
        print("\n=== LCD generation faithfulness (RAGAS, reference-free, disclaimer-excluded) ===")
        for k, v in evaluate_faithfulness(strip_disclaimer=True).items():
            print(f"  {k}: {v}")
