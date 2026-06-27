"""Paired old-vs-new gap eval on a seeded-random 25-record sample.

Runs the CURRENT working-tree pipeline (disambiguation + tightened Partial +
Boundary Check) on 25 random golden records, and compares against the OLD
whole-NCD pipeline's results on the SAME 25 records (from the saved samples
file). Structural metrics only (faithfulness skipped — fast/cheap check).
"""
import collections
import json
import random
from pathlib import Path

SEED, N = 42, 25
SAMPLES = Path("logs/eval_gap_samples_latest.json")
CACHE = Path("logs/gap_answers_cache_k10_t0.65_n5_pk12.json")


def metrics(rows):
    n = len(rows)
    acc = sum(r["alignment_accuracy"] for r in rows) / n
    lm = sum(r["alignment_label_match"] for r in rows) / n
    nr = sum(r["ncd_recall"] for r in rows) / n
    return acc, lm, nr


def per_label(rows):
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ref = r["reference_alignment"]
        by[ref][1] += 1
        by[ref][0] += 1 if r["alignment_label_match"] else 0
    return by


def main():
    # 1) snapshot OLD samples before evaluate() overwrites them
    old = {r["question"]: r for r in json.loads(SAMPLES.read_text(encoding="utf-8"))}

    # 2) the exact seeded-25 questions (from current golden = 276)
    golden = json.loads(Path("data/golden_gap.json").read_text(encoding="utf-8"))
    sampled = random.Random(SEED).sample(golden, N)
    qs = [r["question"] for r in sampled]

    # 3) clear stale cache so the NEW pipeline regenerates
    if CACHE.exists():
        CACHE.unlink()
        print("cleared stale answer cache", flush=True)

    # 4) run NEW pipeline on the seeded-25 (metrics recomputed from the samples
    #    file below for the paired comparison; the returned aggregate is unused)
    from src.evaluation.judge_gap import evaluate
    evaluate(n_samples=N, sample_seed=SEED, faithfulness_max=0)

    # 5) load NEW per-record samples and pair with OLD on the same questions
    new = {r["question"]: r for r in json.loads(SAMPLES.read_text(encoding="utf-8"))}
    old_rows = [old[q] for q in qs if q in old]
    new_rows = [new[q] for q in qs if q in new]

    print(f"\n=== paired old-vs-new on {len(new_rows)} random records (seed={SEED}) ===")
    if old_rows:
        oa, olm, onr = metrics(old_rows)
        print(f"OLD whole-NCD : acc {oa:.3f} | label_match {olm:.3f} | ncd_recall {onr:.3f}  (n={len(old_rows)})")
    na, nlm, nnr = metrics(new_rows)
    print(f"NEW pipeline  : acc {na:.3f} | label_match {nlm:.3f} | ncd_recall {nnr:.3f}  (n={len(new_rows)})")

    print("\nper-label label_match (OLD -> NEW):")
    ob, nb = per_label(old_rows), per_label(new_rows)
    for lab in sorted(set(ob) | set(nb)):
        oh, ot = ob.get(lab, [0, 0])
        nh, nt = nb.get(lab, [0, 0])
        os_ = f"{oh}/{ot}={oh/ot:.2f}" if ot else "-"
        ns_ = f"{nh}/{nt}={nh/nt:.2f}" if nt else "-"
        print(f"  {lab:<24} {os_:>12}  ->  {ns_:>12}")

    print("\nflips (ref | OLD -> NEW):")
    for q in qs:
        if q in old and q in new and old[q]["actual_alignment"] != new[q]["actual_alignment"]:
            o, nw = old[q], new[q]
            ncd = next((r["expected_ncd"] for r in sampled if r["question"] == q), "?")
            print(f"  {ncd:<10} {o['reference_alignment']:<22} {o['actual_alignment']:<22} -> {nw['actual_alignment']}")


if __name__ == "__main__":
    main()
