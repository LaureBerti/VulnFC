#!/usr/bin/env python3
"""Score the recurrent attack-sophistication grid as Detection Failure Rate.

Both attacks in this grid are label-changing:

    llm_generated   LABEL_CHANGING
    pos_negation    LABEL_CHANGING

so both are scored against the rewritten claim's own label, and the quantity is DFR.

The conversion from a stored Vulnerability Score is exact:

    DFR = 1 - VS

holds identically over one fixed set of predictions. The two rates share the denominator
|{c in T : f(c) = l*}|; VS counts the targets whose post-attack answer differs from l*, DFR
counts those whose answer still equals l*, and on a binary task those two sets partition the
denominator. The recurrent victims are deterministic classifiers with no unavailable
predictions, so the partition is complete and no row is lost.

This is therefore a re-scoring, not a re-measurement: no model is loaded, no attack is
generated, and the output is reproducible from the released CSV alone. The identity applies
only within one execution. Between two executions it does not hold, because regenerating an
attack or retraining a victim changes both the predictions and the scored item set.

Usage:
    python experiments/rescore_sophistication.py
    python experiments/rescore_sophistication.py --csv outputs/c9_v2/c9_v2_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from attack_semantics import LABEL_CHANGING

CORPORA = ["kaggle", "politifact", "covid19", "pubhealth", "climatefever", "tiger"]
MODELS = ["lstm_title", "bi_lstm", "rnn"]
ATTACKS = ("llm_generated", "pos_negation")
SEEDS_PER_CELL = 3


def seed_averaged(rows: list[dict]) -> dict:
    """Mean VS per (corpus, victim, attack), over that cell's seeds."""
    acc: dict = defaultdict(list)
    for r in rows:
        acc[(r["dataset"], r["model"], r["attack_type"])].append(float(r["vs"]))
    sizes = {len(v) for v in acc.values()}
    if sizes != {SEEDS_PER_CELL}:
        print(f"[rescore] WARNING: cells hold {sorted(sizes)} rows, expected {SEEDS_PER_CELL}",
              file=sys.stderr)
    return {k: st.mean(v) for k, v in acc.items()}


def ci95(xs: list[float]) -> tuple[float, float, float]:
    m = st.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = 1.96 * st.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/c9_v2/c9_v2_results.csv")
    args = ap.parse_args()

    for a in ATTACKS:
        if a not in LABEL_CHANGING:
            raise SystemExit(f"[rescore] {a} is not label-changing; this script does not apply")

    path = ROOT / args.csv
    if not path.exists():
        raise SystemExit(f"[rescore] not found: {path}")
    rows = list(csv.DictReader(open(path)))

    vs = seed_averaged(rows)
    dfr = {k: 1.0 - v for k, v in vs.items()}

    pairs = []
    for d in CORPORA:
        for m in MODELS:
            if (d, m, ATTACKS[0]) not in dfr or (d, m, ATTACKS[1]) not in dfr:
                continue
            pairs.append((d, m, dfr[(d, m, ATTACKS[0])], dfr[(d, m, ATTACKS[1])]))

    deltas = [L - N for _, _, L, N in pairs]
    n = len(deltas)
    m, lo, hi = ci95(deltas)

    print(f"{len(rows)} rows -> {n} seed-averaged cells, metric DFR\n")
    print(f"{'corpus':13s} {'victim':11s} {'L (DFR)':>8s} {'N (DFR)':>8s} {'L-N':>8s}")
    for d, mo, L, N in pairs:
        print(f"{d:13s} {mo:11s} {L:8.3f} {N:8.3f} {L-N:+8.3f}")

    print(f"\nall {n} cells")
    print(f"  rule-based negation stronger : {sum(1 for x in deltas if x < 0)}")
    print(f"  LLM-generated stronger       : {sum(1 for x in deltas if x > 0)}")
    print(f"  tied                         : {sum(1 for x in deltas if x == 0)}"
          f"   at {sorted({round(L,3) for _,_,L,N in pairs if L == N}) or '--'}")
    print(f"  mean delta (L-N) = {m:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    nz = [x for x in deltas if x != 0]
    if nz:
        print(f"  over the {len(nz)} non-tied cells: mean {st.mean(nz):+.3f}")

    worst = max(abs((1.0 - vs[k]) - dfr[k]) for k in vs)
    print(f"\nmax |DFR - (1 - VS)| over all cells = {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
