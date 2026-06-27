"""Disambiguation-step backtest for primary-NCD selection (cap-2 output).

When the top reranked NCDs are close, one bounded LLM call reads the question +
the top-N candidate NCD titles and picks which 1-2 actually govern. Unlike the
score threshold and the scope rule, this targets the *semantic* signal — and by
considering the top-N (not just top-2) it can reach rank-3+ expected NCDs that
cap-2 selection structurally cannot.

Reuses the cached rerank aggregates (logs/ncd_select_aggs.json) so only the LLM
calls cost anything. Fires the LLM only on ambiguous cases (runner-up within
MARGIN of the top); unambiguous cases keep the deterministic top-1.

Metric defs match the other backtests: top1 / recall (expected-in-selected) /
multi% (two NCDs selected) and llm_calls.
"""
import json
import re
import sys
import time
from pathlib import Path

from src.rag.pipeline import _get_db

AGGS_CACHE = Path("logs/ncd_select_aggs.json")
MARGIN = 0.40   # fire LLM when runner-up score >= MARGIN * top score
TOP_N = 4       # candidates shown to the LLM


def title_map() -> dict[str, str]:
    got = _get_db().get(include=["metadatas"])
    m: dict[str, str] = {}
    for md in got["metadatas"]:
        n, t = md.get("policy_number"), md.get("title")
        if n and n not in m:
            m[n] = t or ""
    return m


def _llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


_SYS = (
    "You identify which Medicare National Coverage Determination(s) govern a coverage "
    "question. You are given the question and a short list of candidate NCDs (number + "
    "title). Return ONLY the NCD number(s) whose policy actually governs the question, "
    "as a JSON list of strings. Prefer a SINGLE NCD. Return two ONLY when the question "
    "genuinely spans a policy and its sub-policy (e.g. a procedure NCD plus its testing/"
    "device sub-NCD). Never return more than two. Use only numbers from the candidate "
    "list. Example: [\"240.4\"] or [\"240.4\",\"240.4.1\"]."
)


def disambiguate(llm, question: str, cands: list[tuple[str, str]]) -> list[str]:
    listing = "\n".join(f'- {n}: {t}' for n, t in cands)
    prompt = f"{_SYS}\n\nQUESTION: {question}\n\nCANDIDATES:\n{listing}\n\nGoverning NCD number(s) as JSON:"
    for attempt in range(6):
        try:
            txt = llm.invoke(prompt).content
            break
        except Exception as exc:
            s = str(exc)
            if attempt == 5 or not any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                raise
            time.sleep(min(2 ** attempt * 5 + 1, 60))
    nums = re.findall(r'\d+(?:\.\d+)+', txt)
    valid = {n for n, _ in cands}
    out: list[str] = []
    for n in nums:
        if n in valid and n not in out:
            out.append(n)
        if len(out) == 2:
            break
    return out or [cands[0][0]]


def main() -> None:
    if not AGGS_CACHE.exists():
        print("Run scripts.backtest_scope_select first to build the aggregate cache.", file=sys.stderr)
        sys.exit(1)
    aggs = [(e, a) for e, a in json.loads(AGGS_CACHE.read_text(encoding="utf-8"))]
    titles = title_map()
    llm = _llm()

    total = len(aggs)
    top1 = inset = multi = calls = 0
    for i, (exp, agg) in enumerate(aggs, 1):
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            continue
        top_n, top_s = ranked[0]
        if len(ranked) > 1 and top_s > 0 and ranked[1][1] >= MARGIN * top_s:
            cands = [(n, titles.get(n, "")) for n, _ in ranked[:TOP_N]]
            chosen = disambiguate(llm, QUESTIONS[i - 1], cands)
            calls += 1
        else:
            chosen = [top_n]
        if chosen and chosen[0] == exp:
            top1 += 1
        if exp in chosen:
            inset += 1
        if len(chosen) > 1:
            multi += 1
        if i % 25 == 0:
            print(f"  ...{i}/{total} (llm calls so far: {calls})", file=sys.stderr)

    print(f"\n=== disambiguation backtest (n={total}, cap=2, margin={MARGIN}, top_n={TOP_N}) ===")
    print(f"llm_calls : {calls}")
    print(f"top1      : {top1/total:.3f}")
    print(f"recall    : {inset/total:.3f}")
    print(f"multi%    : {multi/total:.3f}")


# questions are aligned to the golden order the aggs were built from
QUESTIONS = [r["question"] for r in json.loads(Path("data/golden_gap.json").read_text(encoding="utf-8"))]


if __name__ == "__main__":
    main()
