"""Scope-aware primary-NCD selection backtest (LLM-free, cap held at 2).

Compares the blunt `second_frac` threshold against *scope-aware* selection that
admits the runner-up only when it is hierarchically related to the top NCD
(shared dotted-prefix >= 2 components: parent/child/sibling), rejecting chapter-
mates and unrelated strays. The expensive rerank aggregates are cached to
`logs/ncd_select_aggs.json` so strategies can be re-compared without re-reranking.

Metric definitions match scripts/sweep_second_frac.py:
  top1   — expected NCD is the score-weighted #1
  recall — expected NCD is in the selected set (expected-in-selected)
  multi% — a runner-up was admitted (cross-policy contamination proxy)
"""
import json
import math
import sys
from pathlib import Path

from src.rag.pipeline import PIPELINE_CONFIG

AGGS_CACHE = Path("logs/ncd_select_aggs.json")


def related(a: str, b: str) -> bool:
    """True if two NCD numbers share a dotted-prefix of >= 2 components
    (parent/child or sibling under a real sub-chapter), e.g. 240.4 ~ 240.4.1,
    30.3.1 ~ 30.3.2. Chapter-mates that share only one component (110.3 vs 110.9)
    are NOT related."""
    pa, pb = a.split("."), b.split(".")
    common = 0
    for x, y in zip(pa, pb):
        if x == y:
            common += 1
        else:
            break
    return common >= 2


def ranked_of(agg: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


# ── selection strategies: each returns the chosen NCD list (cap 2) ──────────────
def sel_frac(agg, frac):
    r = ranked_of(agg)
    if not r:
        return []
    chosen = [r[0][0]]
    if len(r) > 1 and r[0][1] > 0 and r[1][1] >= frac * r[0][1]:
        chosen.append(r[1][0])
    return chosen


def sel_scope(agg):
    """Admit runner-up iff hierarchically related to the top NCD (no score gate)."""
    r = ranked_of(agg)
    if not r:
        return []
    chosen = [r[0][0]]
    if len(r) > 1 and related(r[0][0], r[1][0]):
        chosen.append(r[1][0])
    return chosen


def sel_scope_or_frac(agg, frac):
    """Admit runner-up iff related OR score within a tight `frac` (genuine
    co-governance even when not hierarchically named)."""
    r = ranked_of(agg)
    if not r:
        return []
    chosen = [r[0][0]]
    if len(r) > 1 and r[0][1] > 0:
        rn, rs = r[1]
        if related(r[0][0], rn) or rs >= frac * r[0][1]:
            chosen.append(rn)
    return chosen


def build_aggs() -> list[tuple[str, dict[str, float]]]:
    if AGGS_CACHE.exists():
        print(f"Loading cached aggregates: {AGGS_CACHE}", file=sys.stderr)
        raw = json.loads(AGGS_CACHE.read_text(encoding="utf-8"))
        return [(e, a) for e, a in raw]

    from src.rag.pipeline import _hybrid_retrieve_ncd, _rerank_scored

    gold = json.loads(Path("data/golden_gap.json").read_text(encoding="utf-8"))
    k, top_n = PIPELINE_CONFIG["k"], PIPELINE_CONFIG["reranker_top_n"]
    aggs: list[tuple[str, dict[str, float]]] = []
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
    AGGS_CACHE.write_text(json.dumps(aggs), encoding="utf-8")
    print(f"Cached aggregates -> {AGGS_CACHE}", file=sys.stderr)
    return aggs


def score(aggs, selector) -> tuple[float, float, float]:
    top1 = inset = multi = 0
    total = len(aggs)
    for exp, agg in aggs:
        pri = selector(agg)
        if pri and pri[0] == exp:
            top1 += 1
        if exp in pri:
            inset += 1
        if len(pri) > 1:
            multi += 1
    return top1 / total, inset / total, multi / total


def main() -> None:
    aggs = build_aggs()
    total = len(aggs)

    # Diagnostic: of cases where expected is the runner-up, how many are
    # hierarchically related to top-1? (the clean ceiling of scope-aware recall)
    rel = unrel = 0
    for exp, agg in aggs:
        r = ranked_of(agg)
        if len(r) > 1 and r[1][0] == exp and r[0][0] != exp:
            if related(r[0][0], exp):
                rel += 1
            else:
                unrel += 1
    print(f"\n=== runner-up = expected NCD: {rel} related / {unrel} unrelated ===")

    strategies = [
        ("frac=0.70 (current)", lambda a: sel_frac(a, 0.70)),
        ("frac=0.50 (loosest)", lambda a: sel_frac(a, 0.50)),
        ("scope-only", sel_scope),
        ("scope OR frac=0.85", lambda a: sel_scope_or_frac(a, 0.85)),
        ("scope OR frac=0.75", lambda a: sel_scope_or_frac(a, 0.75)),
    ]
    print(f"\n=== strategy comparison (n={total}, cap=2) ===")
    print(f"{'strategy':<22} {'top1':>8} {'recall':>8} {'multi%':>8}")
    for name, sel in strategies:
        t1, rc, mu = score(aggs, sel)
        print(f"{name:<22} {t1:>8.3f} {rc:>8.3f} {mu:>8.3f}")


if __name__ == "__main__":
    main()
