"""LLM-free backtest of NCD selection: does the new score-weighted primary-NCD
picker choose the governing NCD, vs the old 'all NCDs in the top-5 chunks' set?

Runs retrieve -> rerank -> select only (no Gemini), so it's cheap and isolates
retrieval/selection accuracy against the golden gap dataset's expected_ncd.
"""
import json
import random
import sys
from pathlib import Path

from src.rag.pipeline import (
    PIPELINE_CONFIG,
    _hybrid_retrieve_ncd,
    _rerank_scored,
    _select_primary_ncds,
)

GOLDEN = Path("data/golden_gap.json")


def main(n_samples: int | None) -> None:
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if n_samples:
        gold = random.Random(42).sample(gold, min(n_samples, len(gold)))
    k = PIPELINE_CONFIG["k"]
    top_n = PIPELINE_CONFIG["reranker_top_n"]

    new_top1 = new_inset = old_inset = 0
    multi = 0          # times the escape hatch picked a 2nd NCD
    dropped = []       # OLD had expected, NEW dropped it
    rescued = []       # NEW top-1 correct where OLD set was ambiguous (>1 NCD)
    total = len(gold)

    for i, rec in enumerate(gold, 1):
        q = rec["question"]
        exp = rec["expected_ncd"]
        reranked = _rerank_scored(q, _hybrid_retrieve_ncd(q, k), top_n=top_n)
        old_set = {d.metadata.get("policy_number") for _, d in reranked if d.metadata.get("policy_number")}
        primary = _select_primary_ncds(reranked)

        if primary and primary[0] == exp:
            new_top1 += 1
        if exp in primary:
            new_inset += 1
        if exp in old_set:
            old_inset += 1
        if len(primary) > 1:
            multi += 1
        if exp in old_set and exp not in primary:
            dropped.append((exp, primary, q[:60]))
        if primary and primary[0] == exp and len(old_set) > 1:
            rescued.append(exp)
        if i % 25 == 0:
            print(f"  ...{i}/{total}", file=sys.stderr)

    print(f"\n=== NCD selection backtest (n={total}) ===")
    print(f"NEW primary top-1 accuracy : {new_top1}/{total} = {new_top1/total:.3f}")
    print(f"NEW expected-in-selected   : {new_inset}/{total} = {new_inset/total:.3f}")
    print(f"OLD expected-in-top5-chunks: {old_inset}/{total} = {old_inset/total:.3f}")
    print(f"escape hatch picked 2 NCDs : {multi}/{total} = {multi/total:.3f}")
    print(f"NEW dropped an expected NCD OLD had: {len(dropped)}")
    for exp, pri, q in dropped[:15]:
        print(f"   expected {exp:>8} -> picked {pri} | {q}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
