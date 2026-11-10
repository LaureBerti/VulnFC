#!/usr/bin/env python3
"""Measure train/test claim overlap for every (corpus, seed) the benchmark uses.

The learnability gate tests whether a victim learned its task, by lift over the majority-class
baseline and prediction skew. It cannot see memorisation: when test claims are present in the
training split, high accuracy and low skew are exactly what a memorising victim produces, so such a
victim passes the gate with the largest lift in the grid.

Leakage is a property of the (corpus, seed) split, not of the victim, so it is measured once per
split and applies to every victim trained on it.

The predicate is the one the retraining harness enforces, replicated exactly: the test split is
shuffled with `random.Random(seed)` and truncated to `--n-test` rows, then overlap is exact
claim-string membership in the training split. The shuffle selects which rows are scored, so it is
part of the predicate rather than a presentation detail.

Outputs `outputs/split_leakage.csv` with one row per (corpus, seed).

Usage:
    python experiments/check_split_leakage.py
    python experiments/check_split_leakage.py --write
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

CORPORA = ["kaggle", "covid19", "pubhealth", "politifact", "climatefever", "tiger"]
SEEDS = [42, 123, 456]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-test", type=int, default=400,
                    help="rows of the test split the harness scores")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from data_loaders import load_dataset_split

    rows, unchecked = [], []
    print(f"TRAIN/TEST CLAIM OVERLAP  (over the first {args.n_test} test rows)")
    print(f"  {'corpus':13} {'seed':>5} {'n_train':>8} {'n_test':>7} {'overlap':>8} {'frac':>7}  verdict")
    for corpus in args.corpora:
        for seed in args.seeds:
            try:
                train, _, test = load_dataset_split(corpus, seed=seed)
            except Exception as e:
                unchecked.append((corpus, seed, type(e).__name__))
                print(f"  {corpus:13} {seed:>5}  UNCHECKED -- could not load ({type(e).__name__})")
                continue
            rng = random.Random(seed)
            rng.shuffle(test)
            scored = test[: args.n_test]
            train_claims = {r["claim"] for r in train}
            overlap = sum(1 for r in scored if r["claim"] in train_claims)
            frac = overlap / len(scored) if scored else 0.0
            verdict = "CLEAN" if overlap == 0 else "LEAK"
            rows.append({"dataset": corpus, "seed": seed, "n_train": len(train),
                         "n_test_scored": len(scored), "overlap": overlap,
                         "overlap_frac": round(frac, 4), "verdict": verdict})
            print(f"  {corpus:13} {seed:>5} {len(train):>8} {len(scored):>7} "
                  f"{overlap:>8} {frac:>7.4f}  {verdict}")

    leaking = sorted({r["dataset"] for r in rows if r["verdict"] == "LEAK"})
    clean = sorted({r["dataset"] for r in rows} - set(leaking))
    print(f"\n  corpora with any leaking seed: {leaking or 'none'}")
    print(f"  corpora clean at every seed:   {clean or 'none'}")
    print("\n  A leaking split disqualifies every victim trained on it, whatever the gate says.")

    if unchecked:
        print(f"\n  {len(unchecked)} (corpus, seed) pair(s) COULD NOT BE CHECKED:")
        for c, s, err in unchecked:
            print(f"    {c} seed {s}: {err}")
        print("  A verdict of 'no leakage' does NOT cover these. Fetch the corpora and re-run.")

    if args.write and rows:
        out = ROOT / "outputs/split_leakage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 2 if unchecked else 0


if __name__ == "__main__":
    raise SystemExit(main())
