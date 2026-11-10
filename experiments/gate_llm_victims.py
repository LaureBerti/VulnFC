#!/usr/bin/env python3
"""Apply the learnability gate to the LLM victims, on a population matched to its own baseline.

The gate is applied to the recurrent and encoder tracks but never to the LLM victims, and the
released per-condition results do not record the class balance of the subset each condition scored.
A gate verdict therefore cannot be recovered from them: the victim's accuracy comes from a scored
subset of a few hundred claims while any majority-class baseline computed from the full test split
describes a different population. Comparing the two is an unmatched comparison and gives a number
whose error is unknown.

This script closes that gap by measuring both quantities on one population. For each corpus it
draws a sample from the same test split the harness uses, scores the victim on every claim in that
sample through the harness's own prediction path, and computes the baseline, the lift and the
prediction skew over exactly those claims.

The victim is zero-shot, so there is nothing to train and no train/test folds to rotate. Folds are
used only to put an interval on the lift: the sample is partitioned k ways, the lift is computed
within each fold against that fold's own baseline, and the spread across folds gives the uncertainty
the gate verdict needs. A verdict that flips inside the fold interval is reported as indeterminate
rather than as a pass or a fail.

Outputs outputs/post_paper/gate_llm_victims.csv, one row per (corpus, victim, shot mode).

Usage:
    python experiments/gate_llm_victims.py --n-claims 1000
    python experiments/gate_llm_victims.py --corpora covid19 --n-claims 400 --write
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from make_tables import GATE_MAX_SKEW, GATE_MIN_LIFT

CORPORA = ["kaggle", "covid19", "politifact", "pubhealth"]


def folds(items: list, k: int) -> list[list]:
    return [items[i::k] for i in range(k)]


def gate_on(preds: list[int], labels: list[int]) -> dict:
    """Accuracy, majority baseline, lift and skew over ONE population."""
    n = len(labels)
    acc = sum(1 for p, l in zip(preds, labels) if p == l) / n
    maj = max(Counter(labels).values()) / n
    skew = max(Counter(preds).values()) / n
    return {"n": n, "accuracy": acc, "majority": maj, "lift": acc - maj, "skew": skew}


def append_row(out: Path, row: dict) -> None:
    """Append one corpus's row as soon as that corpus finishes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--seed", type=int, default=42, help="the split seed the harness uses")
    ap.add_argument("--n-claims", type=int, default=1000,
                    help="claims sampled from the test split; the whole split if smaller")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--victim-model", default="llama3.2:latest")
    ap.add_argument("--shot-mode", default="zero_shot")
    ap.add_argument("--api-base", default="http://localhost:11434/v1")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent scoring requests; keep at or below the server's "
                         "parallel slots so requests are not queued behind one another")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from data_loaders import load_dataset_split
    from eval_llm_victim import APIVictim
    from victim_api import is_victim_spec, load_victim

    if is_victim_spec(args.victim_model):
        victim = load_victim(args.victim_model)
    else:
        victim = APIVictim(args.victim_model, args.api_base, None, None)
    print(f"LEARNABILITY GATE, LLM VICTIMS  (lift > {GATE_MIN_LIFT}, skew < {GATE_MAX_SKEW})")
    print(f"  victim={args.victim_model}  shot={args.shot_mode}  split seed={args.seed}  "
          f"n<={args.n_claims}  folds={args.k_folds}")
    print("  accuracy, baseline and skew are all computed over the SAME sampled claims.\n")

    OUT = ROOT / "outputs/post_paper/gate_llm_victims.csv"
    rows, unchecked = [], []
    for corpus in args.corpora:
        try:
            _, _, test = load_dataset_split(corpus, seed=args.seed)
        except Exception as e:
            unchecked.append((corpus, type(e).__name__))
            print(f"  {corpus:12} UNCHECKED — could not load ({type(e).__name__})")
            continue
        rng = random.Random(args.seed)
        rng.shuffle(test)
        sample = test[: args.n_claims]

        print(f"  {corpus:12} scoring {len(sample)} claims with {args.workers} worker(s)...",
              flush=True)

        def score(rec: dict) -> tuple:
            return victim.predict(rec["claim"], args.shot_mode), int(rec["label"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            scored = list(ex.map(score, sample))

        preds, labels, n_unavail = [], [], 0
        for p, lab in scored:
            if p is None:
                n_unavail += 1
                continue
            preds.append(int(p))
            labels.append(lab)

        if len(labels) < args.k_folds:
            unchecked.append((corpus, "too few scored claims"))
            print(f"  {corpus:12} UNCHECKED — only {len(labels)} scored")
            continue

        overall = gate_on(preds, labels)
        pl = list(zip(preds, labels))
        lifts = [gate_on([p for p, _ in f], [l for _, l in f])["lift"]
                 for f in folds(pl, args.k_folds) if len(f) >= 2]
        lo, hi = min(lifts), max(lifts)
        sd = st.stdev(lifts) if len(lifts) > 1 else 0.0

        if lo > GATE_MIN_LIFT and overall["skew"] < GATE_MAX_SKEW:
            verdict = "PASS"
        elif hi <= GATE_MIN_LIFT or overall["skew"] >= GATE_MAX_SKEW:
            verdict = "FAIL"
        else:
            verdict = "INDETERMINATE"

        print(f"  {corpus:12} n={overall['n']:5d} (unavail {n_unavail:3d})  "
              f"acc={overall['accuracy']:.4f}  maj={overall['majority']:.4f}  "
              f"lift={overall['lift']:+.4f}  skew={overall['skew']:.3f}")
        print(f"               fold lift range [{lo:+.4f}, {hi:+.4f}]  sd={sd:.4f}  "
              f"-> {verdict}")

        row = {"dataset": corpus, "victim": args.victim_model,
               "shot_mode": args.shot_mode, "split_seed": args.seed,
               "workers": args.workers,
               "n_scored": overall["n"], "n_unavailable": n_unavail,
               "accuracy": round(overall["accuracy"], 4),
               "majority_baseline": round(overall["majority"], 4),
               "lift": round(overall["lift"], 4),
               "skew": round(overall["skew"], 4),
               "fold_lift_min": round(lo, 4), "fold_lift_max": round(hi, 4),
               "fold_lift_sd": round(sd, 4), "k_folds": len(lifts),
               "gate": verdict}
        rows.append(row)
        if args.write:
            append_row(OUT, row)

    if rows:
        npass = sum(1 for r in rows if r["gate"] == "PASS")
        print(f"\n  PASS {npass}/{len(rows)}   "
              f"FAIL {sum(1 for r in rows if r['gate'] == 'FAIL')}/{len(rows)}   "
              f"INDETERMINATE {sum(1 for r in rows if r['gate'] == 'INDETERMINATE')}/{len(rows)}")
        print("  Unavailable claims are reported, not dropped silently: a victim that cannot be")
        print("  queried is not a victim that passed.")

    if unchecked:
        print(f"\n  {len(unchecked)} corpus/corpora UNCHECKED:")
        for c, err in unchecked:
            print(f"    {c}: {err}")
        print("  A verdict of 'measurable' does NOT cover these.")

    if args.write and rows:
        print(f"\nwrote {OUT}")
    return 2 if unchecked else 0


if __name__ == "__main__":
    raise SystemExit(main())
