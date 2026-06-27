"""Evaluate gap_analysis() against the golden gap dataset.

Metrics:
  alignment_accuracy   — END-TO-END: reached the reference alignment AND retrieved the
                         right NCD to reason from (a match on the wrong policy = miss)
  alignment_label_match— diagnostic: raw label agreement vs reference, ignoring retrieval
  ncd_recall           — expected NCD surfaced by RETRIEVAL (in policy_sources metadata),
                         not merely echoed in the report — tests retrieval, not parroting
  pmid_recall          — fraction of reference PMIDs cited (when reference_pmids available)
  citation_precision   — fraction of cited PMIDs that were actually retrieved
  faithfulness         — RAGAS faithfulness of gap report vs policy + pubmed contexts
"""

import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from dotenv import load_dotenv

from src.evaluation.generate_golden_gap import canonical_alignment

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_GAP_PATH = Path(__file__).parents[2] / "data" / "golden_gap.json"
EVAL_GAP_LOG_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_results.jsonl"
EVAL_GAP_SAMPLES_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_samples_latest.json"

JUDGE_MODEL = "gemini-2.5-flash"


def _gap_cache_path() -> Path:
    """Cache path keyed by every config knob that changes retrieval, so a config
    change starts a fresh cache instead of silently reusing stale gap answers."""
    from src.rag.pipeline import PIPELINE_CONFIG
    cfg = PIPELINE_CONFIG
    name = (
        f"gap_answers_cache_k{cfg['k']}_t{cfg['threshold']}"
        f"_n{cfg['reranker_top_n']}_pk{cfg['pubmed_k']}.json"
    )
    return Path(__file__).parents[2] / "logs" / name


def _parse_alignment(text: str) -> str:
    """Extract the Alignment: label from a structured gap report."""
    m = re.search(r"Alignment:\s*\**(.+?)\**(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def _parse_pmids(text: str) -> set[str]:
    return set(re.findall(r"PMID\s*(\d+)", text, re.IGNORECASE))


# Provider-facing action buckets: the end user is denial-prevention / appeals staff,
# so what matters is "do I have evidence-based grounds to appeal, or is it covered?".
# Partial + full Coverage Gap both → appeal; Aligned + Overcoverage both → covered
# (for a provider, Overcoverage just means it's still paid — utilization review is the
# payer's job, not theirs).
_ACTION_BUCKET = {
    "Coverage Gap": "appeal",
    "Partial Coverage Gap": "appeal",
    "Aligned": "covered",
    "Overcoverage": "covered",
    "Insufficient Evidence": "manual-review",
}

# Ordinal "evidence-minus-coverage" direction scale for quadratic-weighted kappa.
# Insufficient Evidence is off this axis and is excluded from kappa.
_DIRECTION_ORDER = ["Overcoverage", "Aligned", "Partial Coverage Gap", "Coverage Gap"]


def _quadratic_weighted_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's quadratic-weighted kappa over the ordinal direction scale.

    pairs: (reference, actual) label tuples; only labels in _DIRECTION_ORDER count.
    Near-miss (one-step) disagreements keep most credit; opposite-direction flips are
    penalized; chance agreement is subtracted.
    """
    idx = {lab: i for i, lab in enumerate(_DIRECTION_ORDER)}
    pairs = [(r, a) for r, a in pairs if r in idx and a in idx]
    n, N = len(_DIRECTION_ORDER), len(pairs)
    if N == 0:
        return None
    O = [[0] * n for _ in range(n)]
    for r, a in pairs:
        O[idx[r]][idx[a]] += 1
    row = [sum(O[i]) for i in range(n)]
    col = [sum(O[i][j] for i in range(n)) for j in range(n)]
    W = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    num = sum(W[i][j] * O[i][j] for i in range(n) for j in range(n))
    den = sum(W[i][j] * row[i] * col[j] / N for i in range(n) for j in range(n))
    return round(1 - num / den, 3) if den else None


def evaluate(
    n_samples: int | None = None,
    faithfulness_max: int | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Run gap analysis eval against the golden gap dataset.

    Args:
        n_samples: Number of golden examples to evaluate (None = all).
        faithfulness_max: Cap RAGAS faithfulness to this many samples (None = all).
            Faithfulness is ~60s/sample (slow); the structural metrics are computed
            over every sample regardless, so cap this to keep large runs tractable.
        sample_seed: If set, take a SEEDED-RANDOM subset of n_samples instead of the
            first n_samples (representative mid-size runs rather than the head slice).

    Returns:
        Dict of metric name → score.
    """
    import pandas as pd
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._faithfulness import Faithfulness

    from src.rag.pipeline import gap_analysis, PIPELINE_CONFIG

    if not GOLDEN_GAP_PATH.exists():
        raise FileNotFoundError(
            f"Golden gap dataset not found at {GOLDEN_GAP_PATH}. "
            "Run `python -m src.evaluation.generate_golden_gap` first."
        )
    golden = json.loads(GOLDEN_GAP_PATH.read_text(encoding="utf-8"))
    if n_samples:
        if sample_seed is not None:
            golden = random.Random(sample_seed).sample(golden, min(n_samples, len(golden)))
        else:
            golden = golden[:n_samples]
    logger.info("Evaluating on %d gap golden records", len(golden))

    cache_path = _gap_cache_path()
    if cache_path.exists():
        cached: list[dict] = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Loaded gap answer cache (%s): %d questions already answered", cache_path.name, len(cached))
    else:
        cached = []

    cached_questions = {item["question"] for item in cached}
    answered_count = 0

    for i, item in enumerate(golden, 1):
        if item["question"] in cached_questions:
            logger.info("  Skipping [%d/%d] (cached): %s", i, len(golden), item["question"][:80])
            continue
        if answered_count > 0:
            time.sleep(1.0)
        logger.info("  Running gap_analysis [%d/%d]: %s", i, len(golden), item["question"][:80])
        result = gap_analysis(item["question"])
        answered_count += 1
        cached.append({
            "question": item["question"],
            "expected_ncd": item["expected_ncd"],
            "reference_alignment": item["reference_alignment"],
            "reference_pmids": item.get("reference_pmids", []),
            "gap_report": result["gap_report"],
            "policy_contexts": [d.page_content for d in result["policy_sources"]],
            "policy_ncds": [
                d.metadata.get("policy_number", "") for d in result["policy_sources"] if d.metadata.get("policy_number")
            ],
            "pubmed_contexts": [d.page_content for d in result["pubmed_sources"]],
            "pubmed_pmids": [
                d.metadata.get("pmid", "") for d in result["pubmed_sources"] if d.metadata.get("pmid")
            ],
        })
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    # ── Structural metrics ─────────────────────────────────────────────────────
    rows = []
    for item in cached:
        report = item["gap_report"]
        # Canonicalize BOTH sides to the rubric vocabulary before comparing. The
        # pipeline emits labels with em-dash descriptors ("Coverage Gap — evidence
        # supports...") while the reference is bare ("Coverage Gap"); an exact-string
        # compare would spuriously fail. canonical_alignment collapses both.
        actual_alignment = canonical_alignment(_parse_alignment(report))
        reference_alignment = canonical_alignment(item.get("reference_alignment", ""))
        pmids_cited = _parse_pmids(report)
        ref_pmids = set(item.get("reference_pmids", []))

        # Raw label agreement vs the INDEPENDENT reference label (diagnostic only).
        alignment_label_match = (
            1.0
            if actual_alignment and reference_alignment
            and actual_alignment == reference_alignment
            else 0.0
        )
        # Retrieval-level: did the expected NCD actually get surfaced into policy_sources?
        # (vs. the old check that just parsed the number out of the generated report,
        # which was ~1.0 always since the NCD is handed to the model and it's told to cite it.)
        retrieved_ncds = set(item.get("policy_ncds", []))
        ncd_recall = 1.0 if item["expected_ncd"] and item["expected_ncd"] in retrieved_ncds else 0.0
        # End-to-end alignment_accuracy: credit ONLY when the pipeline reached the
        # reference alignment AND retrieved the right NCD to reason from. A matching
        # label built on the wrong retrieved policy is a false success, so gating on
        # retrieval makes this score reflect retrieval quality, not just reasoning.
        alignment_accuracy = 1.0 if alignment_label_match and ncd_recall else 0.0
        # Provider-facing action match: did the tool put the case in the right action
        # bucket (appeal / covered / manual-review)? Credits Partial<->Coverage-Gap
        # (both = appeal) and Aligned<->Overcoverage (both = covered).
        ref_act = _ACTION_BUCKET.get(reference_alignment)
        act_act = _ACTION_BUCKET.get(actual_alignment)
        alignment_action_match = 1.0 if ref_act and act_act and ref_act == act_act else 0.0
        pmid_recall = (
            len(pmids_cited & ref_pmids) / len(ref_pmids) if ref_pmids else None
        )
        # citation precision: every PMID cited in the report must have been retrieved.
        # Catches fabricated/hallucinated citations. None when the report cites no PMIDs.
        retrieved_pmids = set(item.get("pubmed_pmids", []))
        citation_precision = (
            len(pmids_cited & retrieved_pmids) / len(pmids_cited) if pmids_cited else None
        )
        # Retrieval-grounded recall: of the reference PMIDs the pipeline ACTUALLY retrieved,
        # how many did it cite? Plain pmid_recall conflates two things — the labeler's
        # reference PMIDs come from a deterministic col.get ordering, while the pipeline
        # retrieves by question-relevance (hybrid + reranker), so a reference PMID may never
        # be surfaced and can't possibly be cited. This variant isolates citation BEHAVIOR
        # from that retrieval-mechanism mismatch. None when no reference PMID was retrieved.
        ref_pmids_retrieved = ref_pmids & retrieved_pmids
        pmid_recall_retrieved = (
            len(pmids_cited & ref_pmids_retrieved) / len(ref_pmids_retrieved)
            if ref_pmids_retrieved else None
        )

        rows.append({
            "question": item["question"],
            "reference_alignment": reference_alignment,
            "actual_alignment": actual_alignment,
            "alignment_accuracy": alignment_accuracy,
            "alignment_label_match": alignment_label_match,
            "alignment_action_match": alignment_action_match,
            "ncd_recall": ncd_recall,
            "pmid_recall": pmid_recall,
            "pmid_recall_retrieved": pmid_recall_retrieved,
            "citation_precision": citation_precision,
        })

    # ── RAGAS faithfulness ─────────────────────────────────────────────────────
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0),
        bypass_n=True,
    )

    def _retry_delay(exc: BaseException) -> float | None:
        m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
        return float(m.group(1)) if m else None

    def _retryable(exc: BaseException) -> bool:
        s = str(exc)
        return bool(_retry_delay(exc)) or any(
            t in s for t in ("429", "RESOURCE_EXHAUSTED", "503", "SERVICE_UNAVAILABLE")
        )

    ragas_items = [item for item in cached if item["policy_contexts"]]
    if faithfulness_max is not None and len(ragas_items) > faithfulness_max:
        # Random (seeded) subsample rather than the first-N slice: faithfulness is a
        # corpus-quality signal, so a representative spread across the dataset is a
        # truer estimate than the alphabetical head — and keeps the slow RAGAS calls
        # capped to minimize API usage. Seeded for reproducibility.
        ragas_items = random.Random(42).sample(ragas_items, faithfulness_max)
    dfs = []
    for i, item in enumerate(ragas_items, 1):
        if i > 1:
            time.sleep(1.0)
        logger.info("  Judging faithfulness [%d/%d]", i, len(ragas_items))
        # Faithfulness over BOTH context sets — the report makes claims about CMS
        # policy AND clinical evidence; checking only policy would miss fabricated
        # findings on the PubMed side (the higher hallucination risk).
        sample = EvaluationDataset(samples=[SingleTurnSample(
            user_input=item["question"],
            response=item["gap_report"],
            retrieved_contexts=item["policy_contexts"] + item["pubmed_contexts"],
        )])
        for attempt in range(10):
            try:
                res = ragas_evaluate(sample, metrics=[Faithfulness(llm=llm)])
                break
            except Exception as exc:
                if not _retryable(exc) or attempt == 9:
                    raise
                delay = _retry_delay(exc)
                wait = (delay + random.uniform(1, 3)) if delay else min(2 ** attempt * 10 + random.uniform(0, 2), 120)
                logger.warning("Rate limited (attempt %d/10) — waiting %.0fs", attempt + 1, wait)
                time.sleep(wait)
        dfs.append(res.to_pandas())

    scores: dict[str, Any] = {}
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        scores["faithfulness"] = round(float(df["faithfulness"].mean(skipna=True)), 3)

    rows_df = pd.DataFrame(rows)
    scores["alignment_accuracy"] = round(float(rows_df["alignment_accuracy"].mean()), 3)
    scores["alignment_label_match"] = round(float(rows_df["alignment_label_match"].mean()), 3)
    # Provider-facing: did it land in the right action bucket (appeal/covered/manual)?
    scores["alignment_action_match"] = round(float(rows_df["alignment_action_match"].mean()), 3)
    # Chance-corrected, adjacency-aware agreement on the evidence-vs-coverage direction
    # (excludes Insufficient Evidence, which is off the ordinal axis).
    scores["alignment_kappa"] = _quadratic_weighted_kappa(
        [(r["reference_alignment"], r["actual_alignment"]) for r in rows]
    )
    scores["ncd_recall"] = round(float(rows_df["ncd_recall"].mean()), 3)
    if rows_df["pmid_recall"].notna().any():
        scores["pmid_recall"] = round(float(rows_df["pmid_recall"].mean(skipna=True)), 3)
    if rows_df["pmid_recall_retrieved"].notna().any():
        scores["pmid_recall_retrieved"] = round(
            float(rows_df["pmid_recall_retrieved"].mean(skipna=True)), 3
        )
    if rows_df["citation_precision"].notna().any():
        scores["citation_precision"] = round(float(rows_df["citation_precision"].mean(skipna=True)), 3)
    scores["n"] = len(cached)

    logger.info("Gap eval scores (%d samples): %s", len(cached), scores)

    EVAL_GAP_LOG_PATH.parent.mkdir(exist_ok=True)
    with EVAL_GAP_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "judge_model": JUDGE_MODEL,
            **PIPELINE_CONFIG,
            **scores,
        }) + "\n")

    EVAL_GAP_SAMPLES_PATH.write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = evaluate()
    for metric, score in results.items():
        print(f"{metric}: {score}")
