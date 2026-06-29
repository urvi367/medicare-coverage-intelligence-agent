"""Sweep SILENT_GATE / LCD_GATE against the LCD eval set (no API cost).

Retrieval + rerank is the expensive part and is independent of the gates, so we
compute each record's NCD/LCD top-sigmoids ONCE, then apply every (silent, lcd)
gate combination instantly. Reports lcd_recall / lcd_precision / disposition per
combo so the soft gates can be picked against ground truth instead of guessed.
"""
import json
import math
from pathlib import Path

from src.rag.pipeline import (
    PIPELINE_CONFIG,
    _denoise,
    _full_ncd_docs,
    _hybrid_retrieve_lcd,
    _hybrid_retrieve_ncd,
    _rerank_scored,
    _select_primary_ncds,
    extract_state,
    ncd_disposition,
    resolve_mac,
)

GOLDEN = Path(__file__).parents[1] / "data" / "golden_lcd.json"
SILENT_GRID = [0.55, 0.58, 0.60, 0.62, 0.65]
LCD_GRID = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _scores_for(rec):
    """Capture the gate-independent scores for one record."""
    q, state = rec["question"], rec["state"]
    top_n = PIPELINE_CONFIG["reranker_top_n"]

    nr = _rerank_scored(q, _hybrid_retrieve_ncd(q, PIPELINE_CONFIG["k"]), top_n=top_n)
    ncd_sig = _sigmoid(nr[0][0]) if nr else 0.0
    ncd_governs = False
    if nr:
        primary = _select_primary_ncds(nr)
        if primary:
            docs = [d for n in primary for d in _full_ncd_docs(n)]
            ncd_governs = ncd_disposition("\n".join(d.page_content for d in docs)) == "governs"

    code = extract_state(q) or extract_state(state or "") or (state.strip().upper() if state else "")
    mac = resolve_mac(code)
    lcd_sig, lcd_id = 0.0, ""
    if mac:
        lq = _denoise(q, code)
        lr = _rerank_scored(lq, _hybrid_retrieve_lcd(lq, PIPELINE_CONFIG["k"], mac), top_n=top_n)
        if lr:
            lcd_sig = _sigmoid(lr[0][0])
            lcd_id = lr[0][1].metadata.get("policy_number", "")
    return {"ncd_sig": ncd_sig, "ncd_governs": ncd_governs, "lcd_sig": lcd_sig,
            "lcd_id": lcd_id, "expected": rec["expected_lcd"]}


def main():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cached = [_scores_for(r) for r in golden]
    n = len(cached)
    print(f"cached scores for {n} records\n")
    print(f"{'silent':>7} {'lcd_gate':>9} {'recall':>7} {'prec':>6} {'lcd':>4} {'ncd':>4} {'none':>5}")
    best = None
    for sg in SILENT_GRID:
        for lg in LCD_GRID:
            lcd = ncd = none = hit = 0
            for c in cached:
                if c["ncd_sig"] >= sg and c["ncd_governs"]:
                    ncd += 1
                elif c["lcd_sig"] >= lg:
                    lcd += 1
                    if c["lcd_id"] == c["expected"]:
                        hit += 1
                else:
                    none += 1
            recall = hit / n
            prec = hit / lcd if lcd else 0.0
            tag = ""
            if sg == 0.60 and lg == 0.55:
                tag = "  <- current"
            if best is None or recall > best[0]:
                best = (recall, prec, sg, lg)
            print(f"{sg:>7.2f} {lg:>9.2f} {recall:>7.3f} {prec:>6.2f} {lcd:>4} {ncd:>4} {none:>5}{tag}")
    print(f"\nbest recall: {best[0]:.3f} (prec {best[1]:.2f}) at silent={best[2]}, lcd_gate={best[3]}")


if __name__ == "__main__":
    main()
