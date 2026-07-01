"""Evaluate gap_analysis() on the LCD gap golden set (data/golden_gap_lcd.json).

The LCD counterpart of judge_gap.py. Same gap_analysis() entry point, but the
governing policy is reached through the NCD→LCD cascade, so each question carries a
`state` (routes to the beneficiary's MAC LCD) and an `expected_lcd`. Because some
LCD-topic services are ALSO NCD-governed, the cascade may legitimately resolve to an
NCD — that is reported as `ncd_intercept`, not scored as an LCD miss.

Metrics:
  disposition          — how questions resolved: lcd / ncd / none
  lcd_recall           — expected LCD surfaced by RETRIEVAL (in policy_sources metadata),
                         not merely echoed in the report — tests retrieval, not parroting
  ncd_intercept        — fraction where an NCD governed instead (national coverage)
  none_rate            — fraction that resolved to no policy
  alignment_accuracy   — END-TO-END: reached the reference alignment AND retrieved the
                         right LCD to reason from (a match on the wrong policy = miss)
  alignment_label_match— diagnostic: raw label agreement vs reference, ignoring retrieval
  alignment_action_match — provider-facing bucket (appeal / covered / manual-review) match
  alignment_kappa      — chance-corrected, adjacency-aware agreement on the direction scale
  pmid_recall          — fraction of reference PMIDs cited (when reference_pmids available)
  pmid_recall_retrieved— citation recall isolated from the retrieval-mechanism mismatch
  citation_precision   — fraction of cited PMIDs that were actually retrieved
  faithfulness         — RAGAS faithfulness of gap report vs policy + pubmed contexts

Usage: python -m src.evaluation.judge_gap_lcd
"""

import collections
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
from src.evaluation.judge_gap import (
    JUDGE_MODEL,
    _ACTION_BUCKET,
    _parse_alignment,
    _parse_pmids,
    _quadratic_weighted_kappa,
)

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_GAP_LCD_PATH = Path(__file__).parents[2] / "data" / "golden_gap_lcd.json"
EVAL_GAP_LCD_LOG_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_lcd_results.jsonl"
EVAL_GAP_LCD_SAMPLES_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_lcd_samples_latest.json"


def _gap_lcd_cache_path() -> Path:
    """Cache path keyed by every config knob that changes retrieval, so a config
    change starts a fresh cache instead of silently reusing stale gap answers."""
    from src.rag.pipeline import PIPELINE_CONFIG
    cfg = PIPELINE_CONFIG
    name = (
        f"gap_lcd_answers_cache_k{cfg['k']}_t{cfg['threshold']}"
        f"_n{cfg['reranker_top_n']}_pk{cfg['pubmed_k']}.json"
    )
    return Path(__file__).parents[2] / "logs" / name


def evaluate(
    n_samples: int | None = None,
    faithfulness_max: int | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Run gap analysis eval against the LCD gap golden dataset.

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

    if not GOLDEN_GAP_LCD_PATH.exists():
        raise FileNotFoundError(
            f"LCD gap golden dataset not found at {GOLDEN_GAP_LCD_PATH}."
        )
    golden = json.loads(GOLDEN_GAP_LCD_PATH.read_text(encoding="utf-8"))
    if n_samples:
        if sample_seed is not None:
            golden = random.Random(sample_seed).sample(golden, min(n_samples, len(golden)))
        else:
            golden = golden[:n_samples]
    logger.info("Evaluating on %d LCD gap golden records", len(golden))

    cache_path = _gap_lcd_cache_path()
    if cache_path.exists():
        cached: list[dict] = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Loaded LCD gap answer cache (%s): %d questions already answered", cache_path.name, len(cached))
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
        # state is required to route the cascade to the beneficiary's MAC LCD.
        result = gap_analysis(item["question"], state=item["state"])
        answered_count += 1
        policy_sources = result["policy_sources"]
        # Disposition inferred from what the cascade actually surfaced: an LCD/NCD chunk's
        # `source` metadata, or "none" when no policy resolved. gap_analysis() doesn't
        # return the resolved source directly, so read it off the retrieved docs.
        disposition = (
            policy_sources[0].metadata.get("source", "").lower()
            if policy_sources else "none"
        )
        cached.append({
            "question": item["question"],
            "expected_lcd": item["expected_lcd"],
            "state": item["state"],
            "reference_alignment": item["reference_alignment"],
            "reference_pmids": item.get("reference_pmids", []),
            "disposition": disposition,
            "gap_report": result["gap_report"],
            "policy_contexts": [d.page_content for d in policy_sources],
            "policy_lcds": [
                d.metadata.get("policy_number", "") for d in policy_sources
                if d.metadata.get("source") == "LCD" and d.metadata.get("policy_number")
            ],
            "pubmed_contexts": [d.page_content for d in result["pubmed_sources"]],
            "pubmed_pmids": [
                d.metadata.get("pmid", "") for d in result["pubmed_sources"] if d.metadata.get("pmid")
            ],
        })
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    # ── Structural metrics ─────────────────────────────────────────────────────
    disp = collections.Counter()
    rows = []
    for item in cached:
        report = item["gap_report"]
        disp[item.get("disposition", "none")] += 1
        # Canonicalize BOTH sides to the rubric vocabulary before comparing (the pipeline
        # emits em-dash descriptors, the reference is bare). See judge_gap for the rationale.
        actual_alignment = canonical_alignment(_parse_alignment(report))
        reference_alignment = canonical_alignment(item.get("reference_alignment", ""))
        pmids_cited = _parse_pmids(report)
        ref_pmids = set(item.get("reference_pmids", []))

        alignment_label_match = (
            1.0
            if actual_alignment and reference_alignment
            and actual_alignment == reference_alignment
            else 0.0
        )
        # Retrieval-level: did the EXPECTED LCD actually get surfaced into policy_sources?
        retrieved_lcds = set(item.get("policy_lcds", []))
        lcd_recall = 1.0 if item["expected_lcd"] and item["expected_lcd"] in retrieved_lcds else 0.0
        # End-to-end: credit only when the pipeline reached the reference alignment AND
        # retrieved the right LCD to reason from (a right label on the wrong policy is a
        # false success). Gating on retrieval ties this to retrieval quality, not parroting.
        alignment_accuracy = 1.0 if alignment_label_match and lcd_recall else 0.0
        # Provider-facing action bucket (appeal / covered / manual-review).
        ref_act = _ACTION_BUCKET.get(reference_alignment)
        act_act = _ACTION_BUCKET.get(actual_alignment)
        alignment_action_match = 1.0 if ref_act and act_act and ref_act == act_act else 0.0
        pmid_recall = (
            len(pmids_cited & ref_pmids) / len(ref_pmids) if ref_pmids else None
        )
        retrieved_pmids = set(item.get("pubmed_pmids", []))
        citation_precision = (
            len(pmids_cited & retrieved_pmids) / len(pmids_cited) if pmids_cited else None
        )
        ref_pmids_retrieved = ref_pmids & retrieved_pmids
        pmid_recall_retrieved = (
            len(pmids_cited & ref_pmids_retrieved) / len(ref_pmids_retrieved)
            if ref_pmids_retrieved else None
        )

        rows.append({
            "question": item["question"],
            "expected_lcd": item["expected_lcd"],
            "disposition": item.get("disposition", "none"),
            "reference_alignment": reference_alignment,
            "actual_alignment": actual_alignment,
            "alignment_accuracy": alignment_accuracy,
            "alignment_label_match": alignment_label_match,
            "alignment_action_match": alignment_action_match,
            "lcd_recall": lcd_recall,
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
        # Seeded representative subsample rather than the head slice; caps the slow calls.
        ragas_items = random.Random(42).sample(ragas_items, faithfulness_max)
    dfs = []
    for i, item in enumerate(ragas_items, 1):
        if i > 1:
            time.sleep(1.0)
        logger.info("  Judging faithfulness [%d/%d]", i, len(ragas_items))
        # Faithfulness over BOTH context sets — the report makes claims about CMS policy
        # AND clinical evidence; checking only policy would miss fabricated PubMed findings.
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

    n = len(cached)
    rows_df = pd.DataFrame(rows)
    scores["disposition"] = dict(disp)
    scores["lcd_recall"] = round(float(rows_df["lcd_recall"].mean()), 3)
    scores["ncd_intercept"] = round(disp.get("ncd", 0) / n, 3) if n else 0.0
    scores["none_rate"] = round(disp.get("none", 0) / n, 3) if n else 0.0
    scores["alignment_accuracy"] = round(float(rows_df["alignment_accuracy"].mean()), 3)
    scores["alignment_label_match"] = round(float(rows_df["alignment_label_match"].mean()), 3)
    scores["alignment_action_match"] = round(float(rows_df["alignment_action_match"].mean()), 3)
    scores["alignment_kappa"] = _quadratic_weighted_kappa(
        [(r["reference_alignment"], r["actual_alignment"]) for r in rows]
    )
    if rows_df["pmid_recall"].notna().any():
        scores["pmid_recall"] = round(float(rows_df["pmid_recall"].mean(skipna=True)), 3)
    if rows_df["pmid_recall_retrieved"].notna().any():
        scores["pmid_recall_retrieved"] = round(
            float(rows_df["pmid_recall_retrieved"].mean(skipna=True)), 3
        )
    if rows_df["citation_precision"].notna().any():
        scores["citation_precision"] = round(float(rows_df["citation_precision"].mean(skipna=True)), 3)
    scores["n"] = n

    logger.info("LCD gap eval scores (%d samples): %s", n, scores)

    EVAL_GAP_LCD_LOG_PATH.parent.mkdir(exist_ok=True)
    with EVAL_GAP_LCD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "judge_model": JUDGE_MODEL,
            **PIPELINE_CONFIG,
            **scores,
        }) + "\n")

    EVAL_GAP_LCD_SAMPLES_PATH.write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = evaluate()
    print("\n=== LCD gap analysis eval ===")
    for metric, score in results.items():
        print(f"{metric}: {score}")
