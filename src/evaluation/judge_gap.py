"""Evaluate gap_analysis() against the golden gap dataset.

Metrics:
  alignment_accuracy — exact match on the Alignment label (Aligned / Gap / Conflicting …)
  ncd_recall         — expected NCD policy number cited in the gap report
  pmid_recall        — fraction of reference PMIDs cited (when reference_pmids available)
  faithfulness       — RAGAS faithfulness of gap report vs policy_sources
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

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_GAP_PATH = Path(__file__).parents[2] / "data" / "golden_gap.json"
EVAL_GAP_LOG_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_results.jsonl"
EVAL_GAP_SAMPLES_PATH = Path(__file__).parents[2] / "logs" / "eval_gap_samples_latest.json"

JUDGE_MODEL = "gemini-2.5-flash"


def _gap_cache_path() -> Path:
    from src.rag.pipeline import PIPELINE_CONFIG
    cfg = PIPELINE_CONFIG
    name = f"gap_answers_cache_k{cfg['k']}_threshold{cfg['threshold']}.json"
    return Path(__file__).parents[2] / "logs" / name


def _parse_alignment(text: str) -> str:
    """Extract the Alignment: label from a structured gap report."""
    m = re.search(r"Alignment:\s*\**(.+?)\**(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def _parse_ncd_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d{2,3}\.\d{1,2}(?:\.\d{1,2})*\b", text))


def _parse_pmids(text: str) -> set[str]:
    return set(re.findall(r"PMID\s*(\d+)", text, re.IGNORECASE))


def evaluate(n_samples: int | None = None) -> dict[str, Any]:
    """Run gap analysis eval against the golden gap dataset.

    Args:
        n_samples: Number of golden examples to evaluate (None = all).

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
            "expected_alignment": item["expected_alignment"],
            "reference_pmids": item.get("reference_pmids", []),
            "gap_report": result["gap_report"],
            "policy_contexts": [d.page_content for d in result["policy_sources"]],
            "pubmed_contexts": [d.page_content for d in result["pubmed_sources"]],
        })
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    # ── Structural metrics ─────────────────────────────────────────────────────
    rows = []
    for item in cached:
        report = item["gap_report"]
        actual_alignment = _parse_alignment(report)
        expected_alignment = item.get("expected_alignment", "")
        ncds_cited = _parse_ncd_numbers(report)
        pmids_cited = _parse_pmids(report)
        ref_pmids = set(item.get("reference_pmids", []))

        alignment_match = (
            1.0
            if actual_alignment and expected_alignment
            and actual_alignment.lower() == expected_alignment.lower()
            else 0.0
        )
        ncd_recall = 1.0 if item["expected_ncd"] and item["expected_ncd"] in ncds_cited else 0.0
        pmid_recall = (
            len(pmids_cited & ref_pmids) / len(ref_pmids) if ref_pmids else None
        )

        rows.append({
            "question": item["question"],
            "expected_alignment": expected_alignment,
            "actual_alignment": actual_alignment,
            "alignment_accuracy": alignment_match,
            "ncd_recall": ncd_recall,
            "pmid_recall": pmid_recall,
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
    dfs = []
    for i, item in enumerate(ragas_items, 1):
        if i > 1:
            time.sleep(1.0)
        logger.info("  Judging faithfulness [%d/%d]", i, len(ragas_items))
        sample = EvaluationDataset(samples=[SingleTurnSample(
            user_input=item["question"],
            response=item["gap_report"],
            retrieved_contexts=item["policy_contexts"],
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
    scores["ncd_recall"] = round(float(rows_df["ncd_recall"].mean()), 3)
    if rows_df["pmid_recall"].notna().any():
        scores["pmid_recall"] = round(float(rows_df["pmid_recall"].mean(skipna=True)), 3)
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
