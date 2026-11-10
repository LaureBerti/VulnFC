#!/usr/bin/env python3
"""Apply the learnability gate to the recurrent victims, and recompute the sophistication contrast.

The gate is the same predicate used for the encoder track (lift over the majority-class baseline
> MIN_LIFT, prediction skew < MAX_SKEW). It needs only a confusion matrix, so it runs from the
saved pre-attack metrics rather than from the checkpoints: no inference, no attack, CPU, seconds.

Scoring the recurrent victims matters because a victim that cannot be flipped scores 0.000 under
every attack, which is indistinguishable from a victim that resisted one. Cells that fail the gate
therefore carry no information about attack strength, in either direction.

Outputs, both under outputs/:
  * gate_recurrent.csv  — one row per (corpus, victim) with accuracy, baseline, lift, skew, verdict
  * the contrast recomputed over gate-passing cells only, printed to stdout

Usage:
    python experiments/gate_recurrent.py
    python experiments/gate_recurrent.py --write      # also write the CSV
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import scipy.stats as sp

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "experiments"))
from make_tables import (GATE_MIN_LIFT, GATE_MAX_SKEW,
                         leaking_corpora, recurrent_gate)

VICTIMS = {"lstm_title": "LSTM$_{title}$", "bi_lstm": "Bi-LSTM", "rnn": "RNN"}
CORPORA = ["kaggle", "politifact", "covid19", "pubhealth", "climatefever", "tiger"]


def gate_rows() -> list[dict]:
    """Flatten the generator's gate verdicts into one row per (corpus, victim).

    The predicate itself lives in make_tables.recurrent_gate(), which is also what marks the
    cells of Table X, so the table and this report cannot disagree.
    """
    g = recurrent_gate()
    return [{"dataset": c, "victim": v, "display": d,
             "accuracy": round(g[(c, v)]["accuracy"], 4),
             "majority_baseline": round(g[(c, v)]["majority"], 4),
             "lift": round(g[(c, v)]["lift"], 4),
             "skew": round(g[(c, v)]["skew"], 4),
             "gate": "PASS" if g[(c, v)]["pass"] else "FAIL"}
            for c in CORPORA for v, d in VICTIMS.items()]


def contrast(cells: list[tuple[float, float]]) -> dict:
    """Paired contrast between the generated attack (L) and the rule-based baseline (N)."""
    L = [a for a, _ in cells]
    N = [b for _, b in cells]
    d = [a - b for a, b in zip(L, N)]
    nz = [x for x in d if x != 0]
    res = {"n": len(d), "wins": sum(1 for x in d if x > 0),
           "ties": sum(1 for x in d if x == 0), "losses": sum(1 for x in d if x < 0),
           "mean_all": st.mean(d), "mean_nontied": st.mean(nz) if nz else float("nan")}
    if len(d) > 1:
        half = sp.t.interval(0.95, len(d) - 1, loc=st.mean(d),
                             scale=st.stdev(d) / len(d) ** 0.5)
        res["ci"] = (half[0], half[1])
    if len(nz) > 1:
        res["p_approx"] = sp.wilcoxon(L, N, method="approx").pvalue
        res["p_exact"] = sp.wilcoxon(L, N, method="exact").pvalue
    return res


TABLE_X = {
    ("kaggle", "lstm_title"): (0.165, 0.045), ("kaggle", "bi_lstm"): (0.000, 0.000),
    ("kaggle", "rnn"): (0.324, 0.073),
    ("politifact", "lstm_title"): (0.171, 0.073), ("politifact", "bi_lstm"): (0.000, 0.000),
    ("politifact", "rnn"): (0.309, 0.308),
    ("covid19", "lstm_title"): (0.212, 0.035), ("covid19", "bi_lstm"): (0.264, 0.035),
    ("covid19", "rnn"): (0.193, 0.043),
    ("pubhealth", "lstm_title"): (0.575, 0.394), ("pubhealth", "bi_lstm"): (0.546, 0.512),
    ("pubhealth", "rnn"): (0.577, 0.433),
    ("climatefever", "lstm_title"): (0.000, 0.000), ("climatefever", "bi_lstm"): (0.000, 0.000),
    ("climatefever", "rnn"): (0.236, 0.098),
    ("tiger", "lstm_title"): (0.234, 0.045), ("tiger", "bi_lstm"): (0.346, 0.213),
    ("tiger", "rnn"): (0.375, 0.126),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write outputs/gate_recurrent.csv")
    args = ap.parse_args()

    rows = gate_rows()
    print(f"LEARNABILITY GATE, RECURRENT TRACK (lift > {GATE_MIN_LIFT}, skew < {GATE_MAX_SKEW})")
    print(f"  {'corpus':13} {'victim':11} {'acc':>6} {'maj':>6} {'lift':>8} {'skew':>6}  gate")
    for r in rows:
        print(f"  {r['dataset']:13} {r['victim']:11} {r['accuracy']:6.3f} "
              f"{r['majority_baseline']:6.3f} {r['lift']:+8.4f} {r['skew']:6.3f}  {r['gate']}")
    npass = sum(1 for r in rows if r["gate"] == "PASS")
    print(f"\n  PASS {npass} / {len(rows)}   FAIL {len(rows) - npass} / {len(rows)}")

    leak = leaking_corpora()
    passing = {(r["dataset"], r["victim"]) for r in rows if r["gate"] == "PASS"}
    admissible = {k for k in passing if k[0] not in leak}
    if leak:
        print(f"\n  corpora excluded for split leakage: "
              + ", ".join(f"{c} ({f:.0%})" for c, f in sorted(leak.items())))
    for label, keep in [("all 18 cells as published", lambda k: True),
                        ("gate-passing cells only", lambda k: k in passing),
                        ("admissible: gate-passing AND leak-free", lambda k: k in admissible)]:
        c = contrast([v for k, v in TABLE_X.items() if keep(k)])
        print(f"\n{label.upper()}  (n = {c['n']})")
        print(f"  generated stronger in {c['wins']}, tied {c['ties']}, baseline stronger {c['losses']}")
        print(f"  mean delta {c['mean_all']:+.4f}   95% CI [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]")
        if "p_exact" in c:
            print(f"  Wilcoxon  exact p = {c['p_exact']:.6f}   normal approx p = {c['p_approx']:.5f}")

    if args.write:
        out = ROOT / "outputs/gate_recurrent.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
