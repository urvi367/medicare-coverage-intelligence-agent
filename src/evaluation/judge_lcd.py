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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
GOLDEN_LCD_PATH = Path(__file__).parents[2] / "data" / "golden_lcd.json"


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scores, misses = evaluate()
    print("\n=== LCD cascade eval ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print("\nmisses (expected -> got | question):")
    for exp, got, q in misses[:20]:
        print(f"  {exp:>8} -> {got:<22} {q}")
