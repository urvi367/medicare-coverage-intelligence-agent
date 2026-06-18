"""Dump NCD policy text + the abstracts the labeler saw (retrieval order) for a
range of golden_gap records, so a human/frontier reviewer can re-judge alignment.

Usage: python scripts/dump_gap_review.py START END   (record indices, 0-based, END exclusive)
Writes to scripts/_gap_review_<START>_<END>.txt
"""
import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.fetch import load_documents  # noqa: E402

MAX_NCD = 8000
MAX_ABS = 2500


def main(start: int, end: int) -> None:
    recs = json.loads((ROOT / "data" / "golden_gap.json").read_text(encoding="utf-8"))
    ncds = {n["policy_number"]: n for n in load_documents("ncd")}
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
    col = client.get_collection("pubmed_evidence")

    out = []
    for i in range(start, min(end, len(recs))):
        r = recs[i]
        ncd_no = r["expected_ncd"]
        ncd = ncds.get(ncd_no, {})
        got = col.get(where={"source_ncd_number": ncd_no}, include=["documents", "metadatas"])
        pairs = list(zip(got["documents"], got["metadatas"]))[:12]

        out.append("=" * 100)
        out.append(f"RECORD #{i}  NCD {ncd_no} — {r['topic']}")
        out.append(f"QUESTION: {r['question']}")
        out.append(f"CURRENT LABEL: {r['reference_alignment']}  | adjudicated: {r.get('adjudicated')}")
        out.append(f"CURRENT RATIONALE: {r['reference_rationale']}")
        out.append(f"CURRENT KEY PMIDS: {r.get('reference_pmids')}  | n_judged: {r.get('n_abstracts_judged')}")
        out.append("-" * 100)
        out.append(f"CMS POLICY TEXT ({ncd_no} — {ncd.get('title','?')}):")
        out.append((ncd.get("text", "MISSING") or "MISSING")[:MAX_NCD])
        out.append("-" * 100)
        out.append(f"ABSTRACTS (retrieval order, {len(pairs)} shown):")
        for text, m in pairs:
            out.append(f"\n[PMID {m.get('pmid')} | {m.get('year')} | {m.get('study_type')} | {m.get('journal')}]")
            out.append((text or "")[:MAX_ABS])
        out.append("")

    dest = ROOT / "scripts" / f"_gap_review_{start}_{end}.txt"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dest} ({len(out)} lines)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
