"""Build and load a ChromaDB vector index from ingested CMS data."""

import logging
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.fetch import load_documents
from src.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parents[2] / "data" / "chroma"

# Clinical ↔ CMS vocabulary bridge.
# Keys are terms as they appear in NCD/LCD text; values are clinical/trade synonyms.
# Appended to each chunk at index time so clinical queries reach policy chunks.
_SYNONYMS: dict[str, list[str]] = {
    # GLP-1 / weight management drugs
    "semaglutide": ["Ozempic", "Wegovy", "Rybelsus", "GLP-1 agonist", "GLP-1"],
    "liraglutide": ["Victoza", "Saxenda", "GLP-1 agonist"],
    "dulaglutide": ["Trulicity", "GLP-1 agonist"],
    "tirzepatide": ["Mounjaro", "Zepbound", "GLP-1 GIP dual agonist"],
    # SGLT2 inhibitors
    "empagliflozin": ["Jardiance", "SGLT2 inhibitor"],
    "dapagliflozin": ["Farxiga", "SGLT2 inhibitor"],
    "canagliflozin": ["Invokana", "SGLT2 inhibitor"],
    # Alzheimer's therapies
    "lecanemab": ["Leqembi", "amyloid therapy", "Alzheimer treatment"],
    "aducanumab": ["Aduhelm", "amyloid therapy", "Alzheimer treatment"],
    "donepezil": ["Aricept", "cholinesterase inhibitor"],
    # MS therapies
    "ocrelizumab": ["Ocrevus", "anti-CD20", "MS treatment"],
    "natalizumab": ["Tysabri", "MS treatment"],
    # Cardiac devices
    "implantable cardioverter defibrillator": ["ICD", "defibrillator", "cardiac arrest device"],
    "ventricular assist device": ["VAD", "LVAD", "heart pump", "destination therapy", "bridge to transplant"],
    "cardiac resynchronization therapy": ["CRT", "biventricular pacemaker", "CRT-D", "CRT-P"],
    # Neuromodulation
    "sacral nerve stimulation": ["sacral neuromodulation", "SNS", "overactive bladder", "urinary incontinence", "bladder control"],
    "transcranial magnetic stimulation": ["TMS", "rTMS", "repetitive TMS", "brain stimulation", "depression treatment"],
    "deep brain stimulation": ["DBS", "Parkinson device", "essential tremor treatment"],
    # Glucose / diabetes monitoring
    "continuous glucose monitoring": ["CGM", "Dexcom", "Libre", "Freestyle Libre", "glucose sensor", "iCGM"],
    "blood glucose testing": ["blood sugar test", "glucometer", "fingerstick", "diabetes monitoring", "SMBG"],
    # Imaging
    "positron emission tomography": ["PET scan", "FDG PET", "FDG-PET", "nuclear medicine scan"],
    "fluorodeoxyglucose": ["FDG", "F-18 FDG", "18F-FDG", "PET tracer"],
    # Rehabilitation
    "cardiac rehabilitation": ["cardiac rehab", "heart attack recovery", "post-MI rehabilitation", "coronary artery disease rehab"],
    "pulmonary rehabilitation": ["pulmonary rehab", "COPD rehab", "lung rehabilitation", "breathing therapy"],
    # Conditions
    "chronic obstructive pulmonary disease": ["COPD", "emphysema", "chronic bronchitis"],
    "end-stage renal disease": ["ESRD", "kidney failure", "renal failure", "dialysis"],
    "amyotrophic lateral sclerosis": ["ALS", "Lou Gehrig disease", "motor neuron disease"],
    "age-related macular degeneration": ["AMD", "macular degeneration", "wet AMD", "dry AMD"],
    "prothrombin time": ["PT test", "PT/INR", "INR", "blood clotting test", "coagulation test"],
    # Photodynamic therapy
    "photodynamic therapy": ["PDT", "light therapy", "verteporfin treatment"],
    "verteporfin": ["Visudyne", "PDT drug", "photosensitizer"],
    # Other procedures / devices
    "electroencephalography": ["EEG", "brain wave test", "seizure monitoring"],
    "thermography": ["thermal imaging", "infrared imaging", "heat imaging"],
    "osteogenic stimulator": ["bone growth stimulator", "electrical bone stimulation", "bone healing device"],
    # Nutrition
    "enteral nutrition": ["tube feeding", "nasogastric tube", "PEG tube", "enteral feeding"],
    "parenteral nutrition": ["TPN", "total parenteral nutrition", "IV nutrition", "intravenous feeding"],
    # Oxygen
    "home oxygen": ["supplemental oxygen", "O2 therapy", "oxygen concentrator", "home O2"],
    # Cardiac / hemodynamic monitoring
    "implantable pulmonary artery pressure sensor": ["IPAPS", "CardioMEMS", "PA pressure sensor", "heart failure hemodynamic monitoring"],
    # Knee / orthopedic
    "collagen meniscus implant": ["CMI", "meniscal regeneration", "knee meniscus repair", "torn meniscus treatment"],
    # GI procedures
    "endoscopy": ["endoscopic procedure", "colonoscopy", "upper endoscopy", "EGD", "ERCP", "GI scope"],
}

# Sort longest keys first so multi-word phrases match before substrings.
_SYNONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_SYNONYMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_synonyms(text: str) -> str:
    """Append a synonym line for every CMS term found in text."""
    found: dict[str, list[str]] = {}
    for m in _SYNONYM_RE.finditer(text):
        key = m.group(0).lower()
        if key not in found:
            found[key] = _SYNONYMS[key]
    if not found:
        return text
    syn_line = "; ".join(", ".join(syns) for syns in found.values())
    return text + "\nSynonyms: " + syn_line
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

    # Prepend title to every chunk so clinical and policy terminology in the title
    # is present in every embedded vector, not just the first chunk.
    for chunk in chunks:
        title = chunk.metadata.get("title", "")
        if title:
            chunk.page_content = f"{title}\n{chunk.page_content}"
        chunk.page_content = _expand_synonyms(chunk.page_content)

    before = len(chunks)
    chunks = [c for c in chunks if c.page_content.strip()]
    dropped = before - len(chunks)
    if dropped:
        logger.warning("Dropped %d blank/whitespace-only chunks before indexing", dropped)

    # Chroma.from_documents APPENDS — it assigns fresh IDs rather than replacing.
    # Wipe the persisted index first so rebuilds don't accumulate stale duplicates
    # (e.g. chunks from a previous, pre-fix ingestion).
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Cleared existing index at %s before rebuild", CHROMA_DIR)

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
