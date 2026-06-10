"""RAGAS-based evaluation of the RAG pipeline (faithfulness + answer relevancy)."""

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

# RAGAS 0.4 imports this removed langchain-community module at startup even when unused
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parents[2] / "data" / "golden_dataset.json"
EVAL_LOG_PATH = Path(__file__).parents[2] / "logs" / "eval_results.jsonl"
EVAL_SAMPLES_PATH = Path(__file__).parents[2] / "logs" / "eval_samples_latest.json"

JUDGE_MODEL = "gemini-2.5-flash"
PROMPT_TAG = "v1"


def _answers_cache_path() -> Path:
    """Derive cache path from current PIPELINE_CONFIG so each config gets its own file."""
    from src.rag.pipeline import PIPELINE_CONFIG
    cfg = PIPELINE_CONFIG
    name = f"rag_answers_cache_k{cfg['k']}_threshold{cfg['threshold']}_n{cfg['reranker_top_n']}.json"
    return Path(__file__).parents[2] / "logs" / name


def load_golden_dataset() -> list[dict[str, str]]:
    """Load golden dataset from disk.

    Raises:
        FileNotFoundError: If golden_dataset.json does not exist.
            Run `python -m src.evaluation.generate_golden` to create it.
    """
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {GOLDEN_PATH}. "
            "Run `python -m src.evaluation.generate_golden` to generate it."
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def evaluate(n_samples: int | None = None) -> dict[str, Any]:
    """Run RAGAS faithfulness + answer_relevancy evaluation against the golden dataset.

    Args:
        n_samples: Number of golden examples to evaluate (None = all).

    Returns:
        Dict of metric name → score.
    """
    import pandas as pd
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
    from ragas.metrics._faithfulness import Faithfulness

    from src.rag.pipeline import answer as rag_answer, PIPELINE_CONFIG

    golden = load_golden_dataset()
    if n_samples:
        golden = golden[:n_samples]

    # Filter out LCD entries until jurisdiction handling is implemented.
    # Entries without requires_jurisdiction (pre-dating the field) are assumed NCD-safe.
    pre_filter = len(golden)
    golden = [item for item in golden if not item.get("requires_jurisdiction", False)]
    skipped_jurisdiction = pre_filter - len(golden)
    if skipped_jurisdiction:
        logger.info(
            "Skipped %d/%d entries with requires_jurisdiction=True (LCD jurisdiction handling not yet implemented). "
            "Coverage gap: %.0f%% of dataset excluded.",
            skipped_jurisdiction, pre_filter,
            100 * skipped_jurisdiction / pre_filter,
        )

    # --- persistent answer cache: skip questions already answered in any prior run ---
    answers_cache_path = _answers_cache_path()
    if answers_cache_path.exists():
        cached: list[dict] = json.loads(answers_cache_path.read_text(encoding="utf-8"))
        logger.info("Loaded answer cache (%s): %d questions already answered", answers_cache_path.name, len(cached))
    else:
        cached = []

    cached_questions = {item["user_input"] for item in cached}

    # gemini-2.5-flash paid tier: 1000+ RPM → 1s gap is ample headroom
    _ANSWER_DELAY = 1.0
    answered_count = 0

    for i, item in enumerate(golden, 1):
        if item["question"] in cached_questions:
            logger.info("  Skipping [%d/%d] (cached): %s", i, len(golden), item["question"][:80])
            continue
        if answered_count > 0:
            time.sleep(_ANSWER_DELAY)
        logger.info("  Answering [%d/%d]: %s", i, len(golden), item["question"][:80])
        result = rag_answer(item["question"])
        answered_count += 1
        cached.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [s.page_content for s in result["sources"]],
            "source_policy_numbers": [
                s.metadata["policy_number"]
                for s in result["sources"]
                if s.metadata.get("policy_number")
            ],
            "reference": item["reference_answer"],
        })
        answers_cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    def _extract_policy_numbers(text: str) -> set[str]:
        lcd = set(re.findall(r'\bL\d{4,6}\b', text))
        # NCD decimal format: 20.4, 160.6.1 — 2-digit prefix avoids short false positives
        ncd = set(re.findall(r'\b\d{2,3}\.\d{1,2}(?:\.\d{1,2})*\b', text))
        return lcd | ncd

    def _citation_accuracy(item: dict) -> float:
        resp = item["response"]
        stored = [pn for pn in item.get("source_policy_numbers", []) if pn]
        if stored:
            return 1.0 if any(pn in resp for pn in stored) else 0.0
        # Fallback for old cache entries: regex cross-check on page_content
        ctx_nums = _extract_policy_numbers(" ".join(item["retrieved_contexts"]))
        return 1.0 if (_extract_policy_numbers(resp) & ctx_nums) else 0.0

    # restrict evaluation to NCD golden questions only — exclude legacy LCD cache entries
    ncd_questions = {item["question"] for item in golden}
    # map question → expected policy_number for policy_recall metric
    golden_policy = {item["question"]: item.get("policy_number", "") for item in golden}
    eval_cache = [item for item in cached if item["user_input"] in ncd_questions]
    logger.info("Evaluating %d/%d cached entries (NCD golden set only)", len(eval_cache), len(cached))

    # extra metrics computed over ALL eval_cache (including empty-retrieval items)
    extra_rows = []
    for item in eval_cache:
        expected_pn = golden_policy.get(item["user_input"], "")
        source_pns = item.get("source_policy_numbers", [])
        if not expected_pn:
            policy_recall = None          # no expected policy in golden → skip
        elif not source_pns:
            policy_recall = 0.0           # empty retrieval → wrong NCD
        else:
            policy_recall = 1.0 if expected_pn in source_pns else 0.0
        extra_rows.append({
            "user_input": item["user_input"],
            "empty_retrieval_rate": 1.0 if not item["retrieved_contexts"] else 0.0,
            "citation_accuracy": _citation_accuracy(item),
            "policy_recall": policy_recall,
        })

    # RAGAS runs only on items that actually retrieved documents.
    # Empty-retrieval responses ("I cannot answer") are retrieval failures, not
    # faithfulness failures — including them drags down faithfulness/AR unfairly.
    ragas_items = [item for item in eval_cache if item["retrieved_contexts"]]
    skipped_empty = len(eval_cache) - len(ragas_items)
    if skipped_empty:
        logger.info(
            "Excluding %d empty-retrieval items from RAGAS (counted in empty_retrieval_rate)",
            skipped_empty,
        )

    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"],
            reference=item["reference"],
        )
        for item in ragas_items
    ]

    # bypass_n=True: Gemini ignores n>1 and returns 1 generation; this makes RAGAS
    # send n separate single requests instead of one n=3 request.
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0),
        bypass_n=True,
    )
    # Use the same local model as the vector index — no API quota, no availability issues.
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]

    # gemini-2.5-flash paid tier: 1000+ RPM → 1s between samples is ample headroom
    _JUDGE_DELAY = 1.0

    def _judge_retry_delay(exc: BaseException) -> float | None:
        m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
        return float(m.group(1)) if m else None

    def _judge_retryable(exc: BaseException) -> bool:
        s = str(exc)
        return bool(_judge_retry_delay(exc)) or any(
            t in s for t in ("429", "RESOURCE_EXHAUSTED", "503", "SERVICE_UNAVAILABLE")
        )

    dfs = []
    for i, sample in enumerate(samples, 1):
        if i > 1:
            time.sleep(_JUDGE_DELAY)
        logger.info("  Judging [%d/%d]", i, len(samples))
        single = EvaluationDataset(samples=[sample])
        for attempt in range(10):
            try:
                res = ragas_evaluate(single, metrics=metrics)
                break
            except Exception as exc:
                if not _judge_retryable(exc) or attempt == 9:
                    raise
                delay = _judge_retry_delay(exc)
                wait = (delay + random.uniform(1, 3)) if delay else min(2 ** attempt * 10 + random.uniform(0, 2), 120)
                logger.warning("Judge rate limited (attempt %d/10) — waiting %.0fs", attempt + 1, wait)
                time.sleep(wait)
        dfs.append(res.to_pandas())

    if not dfs:
        logger.warning("No RAGAS samples to evaluate — all items had empty retrieval.")
        scores = {"empty_retrieval_rate": 1.0, "citation_accuracy": 0.0, "policy_recall": 0.0, "ragas_n": 0}
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "judge_model": JUDGE_MODEL, "prompt": PROMPT_TAG, **PIPELINE_CONFIG, **scores}) + "\n")
        return scores
    df = pd.concat(dfs, ignore_index=True)
    # RAGAS metrics: averaged over non-empty-retrieval samples only
    skip = {"user_input", "retrieved_contexts", "response", "reference"}
    scores = {col: round(float(df[col].mean(skipna=True)), 3) for col in df.columns if col not in skip}
    # extra metrics: averaged over ALL eval_cache items (empty retrieval included)
    extra_df = pd.DataFrame(extra_rows)
    scores["empty_retrieval_rate"] = round(float(extra_df["empty_retrieval_rate"].mean()), 3)
    scores["citation_accuracy"] = round(float(extra_df["citation_accuracy"].mean(skipna=True)), 3)
    scores["policy_recall"] = round(float(extra_df["policy_recall"].mean(skipna=True)), 3)
    scores["ragas_n"] = len(ragas_items)
    logger.info("Evaluation scores (RAGAS on %d/%d samples): %s", len(ragas_items), len(eval_cache), scores)

    # merge extra_rows into df for per-sample export (left join keeps only ragas rows)
    df = df.merge(extra_df, on="user_input", how="left")

    EVAL_LOG_PATH.parent.mkdir(exist_ok=True)

    # save per-sample rows (exclude bulky context columns for readability)
    sample_cols = ["user_input", "response"] + [c for c in df.columns if c not in skip]
    EVAL_SAMPLES_PATH.write_text(
        json.dumps(df[sample_cols].to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "judge_model": JUDGE_MODEL, "prompt": PROMPT_TAG, **PIPELINE_CONFIG, **scores}) + "\n")

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = evaluate()
    for metric, score in results.items():
        print(f"{metric}: {score}")
