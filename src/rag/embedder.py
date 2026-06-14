"""Local sentence-transformer embeddings — no API cost."""

import os

# Disable HuggingFace/transformers tqdm progress bars BEFORE the model loads.
# Under Streamlit on Windows, sys.stderr is replaced with a stream whose flush()
# raises OSError [Errno 22]; tqdm's "Loading weights" bar then crashes model load.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:  # newer transformers draws a "Loading weights" bar outside the hub downloader
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except Exception:
    pass

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embeddings model (BAAI/bge-small-en-v1.5)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
