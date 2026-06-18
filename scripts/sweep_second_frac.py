"""Sweep the second_frac escape-hatch threshold for primary-NCD selection.

One rerank pass per question; apply each threshold to the cached sigmoid-aggregated
scores. Reports expected-in-selected recall and the multi-NCD (contamination) rate
at each threshold, so we can pick the recall/cleanliness sweet spot.
"""
import json
import math
import sys
from pathlib import Path

from src.rag.pipeline import (
    PIPELINE_CONFIG,
    _hybrid_retrieve_ncd,
    _rerank_scored,
)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def select(agg: dict[str, float], frac: float) -> list[str]:
    if not agg:
        return []
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    top_n, top_s = ranked[0]
    chosen = [top_n]
    for n, s in ranked[1:]:
        if top_s > 0 and s >= frac * top_s:
            chosen.append(n)
        else:
            break
    return chosen


def main() -> None:
    gold = json.loads(Path("data/golden_gap.json").read_text(encoding="utf-8"))
    k, top_n = PIPELINE_CONFIG["k"], PIPELINE_CONFIG["reranker_top_n"]
    aggs = []
    for i, rec in enumerate(gold, 1):
        reranked = _rerank_scored(rec["question"], _hybrid_retrieve_ncd(rec["question"], k), top_n=top_n)
        agg: dict[str, float] = {}
        for s, d in reranked:
            num = d.metadata.get("policy_number")
            if num:
                agg[num] = agg.get(num, 0.0) + 1.0 / (1.0 + math.exp(-s))
        aggs.append((rec["expected_ncd"], agg))
        if i % 25 == 0:
            print(f"  ...{i}/{len(gold)}", file=sys.stderr)

    total = len(aggs)
    print(f"\n=== second_frac sweep (n={total}) ===")
    print(f"{'frac':>6} {'top1':>8} {'recall':>8} {'multi%':>8}")
    for frac in THRESHOLDS:
        top1 = inset = multi = 0
        for exp, agg in aggs:
            pri = select(agg, frac)
            if pri and pri[0] == exp:
                top1 += 1
            if exp in pri:
                inset += 1
            if len(pri) > 1:
                multi += 1
        print(f"{frac:>6.1f} {top1/total:>8.3f} {inset/total:>8.3f} {multi/total:>8.3f}")


if __name__ == "__main__":
    main()
