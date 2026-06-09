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
ANSWERS_CACHE_PATH = Path(__file__).parents[2] / "logs" / "rag_answers_cache.json"
EVAL_LOG_PATH = Path(__file__).parents[2] / "logs" / "eval_results.jsonl"


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

    from src.rag.pipeline import answer as rag_answer

    golden = load_golden_dataset()
    if n_samples:
        golden = golden[:n_samples]

    # Filter out LCD entries until jurisdiction handling is implemented.
    # Entries pre-dating the document_type field are assumed NCD-safe (requires_jurisdiction=False).
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
    if ANSWERS_CACHE_PATH.exists():
        cached: list[dict] = json.loads(ANSWERS_CACHE_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded answer cache: %d questions already answered", len(cached))
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
            "reference": item["reference_answer"],
        })
        ANSWERS_CACHE_PATH.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    samples = [SingleTurnSample(**item) for item in cached]

    # bypass_n=True: Gemini ignores n>1 and returns 1 generation; this makes RAGAS
    # send n separate single requests instead of one n=3 request.
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0),
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

    # gemini-2.5-flash-lite paid tier: 1000+ RPM → 1s between samples is ample headroom
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

    df = pd.concat(dfs, ignore_index=True)
    skip = {"user_input", "retrieved_contexts", "response", "reference"}
    scores = {col: round(float(df[col].mean(skipna=True)), 3) for col in df.columns if col not in skip}
    logger.info("Evaluation scores: %s", scores)

    # persist scores and clean up checkpoint
    EVAL_LOG_PATH.parent.mkdir(exist_ok=True)
    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **scores}) + "\n")

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = evaluate()
    for metric, score in results.items():
        print(f"{metric}: {score}")
