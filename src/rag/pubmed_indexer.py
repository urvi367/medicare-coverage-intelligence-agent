"""Build and load a ChromaDB vector index from PubMed abstracts."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.ingestion.fetch_pubmed import load_documents as load_pubmed
from src.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parents[2] / "data" / "chroma"
PUBMED_COLLECTION = "pubmed_evidence"


def build_pubmed_index() -> Chroma:
    """Embed PubMed abstracts and persist to a separate ChromaDB collection."""
    raw_docs = load_pubmed()
    if not raw_docs:
        raise RuntimeError("No PubMed data. Run `python -m src.ingestion.fetch_pubmed` first.")

    # Abstracts are ~250 words — no chunking needed, one doc per abstract.
    lc_docs = [
        Document(
            page_content=d["text"],
            metadata={k: v for k, v in d.items() if k != "text"},
        )
        for d in raw_docs
    ]

    # Wipe only the pubmed collection, leaving cms_coverage untouched.
    if CHROMA_DIR.exists():
        existing = Chroma(
            collection_name=PUBMED_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        existing.delete_collection()
        logger.info("Cleared existing %r collection before rebuild", PUBMED_COLLECTION)

    logger.info("Indexing %d PubMed abstracts...", len(lc_docs))
    db = Chroma.from_documents(
        lc_docs,
        get_embeddings(),
        collection_name=PUBMED_COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("PubMed index built: %d docs persisted to %s", len(lc_docs), CHROMA_DIR)
    return db


def load_pubmed_index() -> Chroma:
    """Load the persisted PubMed ChromaDB collection from disk."""
    if not CHROMA_DIR.exists():
        raise RuntimeError("PubMed index not found. Run `python -m src.rag.pubmed_indexer` first.")
    return Chroma(
        collection_name=PUBMED_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_pubmed_index()
