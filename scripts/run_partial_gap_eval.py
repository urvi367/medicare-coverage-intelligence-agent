"""One-off: partial gap-eval re-run under the whole-NCD pipeline (budget ~$0.50).

Regenerates the first N golden gap answers with the current pipeline (the stale
config-keyed cache was deleted first) and judges them. Faithfulness capped to keep
the slow RAGAS calls bounded. Not part of the eval suite — see PRD §15.2.2.
"""
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from src.evaluation.judge_gap import evaluate

N = None  # all 277
FAITHFULNESS_MAX = 22  # seeded-random subsample (RAGAS is ~120s/sample)

t = time.time()
res = evaluate(n_samples=N, faithfulness_max=FAITHFULNESS_MAX)
res["elapsed_sec"] = round(time.time() - t)

out = Path("logs") / "gap_eval_wholeNCD_full.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")

print("\n=== FULL WHOLE-NCD GAP EVAL (n=%s) ===" % (N or "all"))
for k, v in res.items():
    print(f"{k}: {v}")
print(f"\nwrote {out}")
