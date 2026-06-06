"""Build and load a ChromaDB vector index from ingested CMS data."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.fetch import load_documents
from src.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parents[2] / "data" / "chroma"
COLLECTION = "cms_coverage"

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_index() -> Chroma:
    """Load NCD + LCD docs, chunk, embed, and persist to ChromaDB."""
    raw_docs: list[dict] = []
    for doc_type in ("ncd", "lcd"):
        try:
            raw_docs.extend(load_documents(doc_type))
        except FileNotFoundError:
            logger.warning("No %s data — run `python -m src.ingestion.fetch` first", doc_type.upper())

    if not raw_docs:
        raise RuntimeError("No documents to index. Run `python -m src.ingestion.fetch` first.")

    lc_docs = [
        Document(
            page_content=d["text"],
            metadata={k: v for k, v in d.items() if k != "text"},
        )
        for d in raw_docs
    ]
    chunks = _SPLITTER.split_documents(lc_docs)
    logger.info("Indexing %d chunks from %d documents...", len(chunks), len(lc_docs))

    db = Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("Index built and persisted to %s", CHROMA_DIR)
    return db


def load_index() -> Chroma:
    """Load the persisted ChromaDB index from disk."""
    if not CHROMA_DIR.exists():
        raise RuntimeError("Index not found. Run `python -m src.rag.indexer` first.")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index()
