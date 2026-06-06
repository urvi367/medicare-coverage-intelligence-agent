"""RAGAS-based evaluation of the RAG pipeline (faithfulness + answer relevancy)."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# RAGAS 0.4 imports this removed langchain-community module at startup even when unused
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parents[2] / "data" / "golden_dataset.json"


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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness

    from src.rag.pipeline import answer as rag_answer

    golden = load_golden_dataset()
    if n_samples:
        golden = golden[:n_samples]

    def _rag_answer_with_retry(question: str, max_retries: int = 5) -> dict:
        from groq import RateLimitError
        for attempt in range(max_retries):
            try:
                return rag_answer(question)
            except RateLimitError as e:
                retry_after = getattr(e.response, "headers", {}).get("retry-after")
                wait = int(float(retry_after)) + 1 if retry_after else 60 * (attempt + 1)
                logger.warning("Rate limited on RAG call — waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
        raise RuntimeError(f"Exhausted retries for question: {question}")

    # llama-3.3-70b-versatile free tier: 12K TPM, ~800 tok/request → max ~15 req/min → 5s delay
    _EVAL_DELAY = 5.0

    samples = []
    for i, item in enumerate(golden, 1):
        logger.info("  Answering [%d/%d]: %s", i, len(golden), item["question"][:80])
        result = _rag_answer_with_retry(item["question"])
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=[s.page_content for s in result["sources"]],
            reference=item["reference_answer"],
        ))
        time.sleep(_EVAL_DELAY)

    dataset = EvaluationDataset(samples=samples)

    # bypass_n=True: Groq rejects n>1 in a single API call (400 error).
    # This makes RAGAS send n separate requests (each n=1) instead.
    llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.environ["GROQ_API_KEY"]),
        bypass_n=True,
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
    ]

    result = ragas_evaluate(dataset, metrics=metrics)
    df = result.to_pandas()
    skip = {"user_input", "retrieved_contexts", "response", "reference"}
    scores = {col: round(float(df[col].mean(skipna=True)), 3) for col in df.columns if col not in skip}
    logger.info("Evaluation scores: %s", scores)
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = evaluate()
    for metric, score in results.items():
        print(f"{metric}: {score}")
