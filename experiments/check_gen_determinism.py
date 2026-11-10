#!/usr/bin/env python3
"""Test whether LLM attack generation reproduces its text when the seed is held fixed.

LLM-generated attacks pass `seed + idx` to the generating model. A per-request seed fixes the
sampling RNG, not the logits it draws from, so it does not by itself make the generated text
reproducible.

This script measures that directly rather than inferring it from scores. It generates the same
claims twice under the same seed and compares the returned strings, at two worker counts:

  workers = N   the configuration the experiments ran (run_exp_C9_v2.sh defaults to 4), where
                concurrent requests are batched together by the server and the logits of one
                request depend on which others share its batch
  workers = 1   requests issued one at a time, so no co-batching is possible

Concurrency is one contributor and not the only one: the text diverges at both worker counts, so a
serial generation pass narrows the variance without removing it.

Requires a running generator (`ollama serve`) and, for the parallel arm, OLLAMA_NUM_PARALLEL >= the
worker count.

Usage:
    python experiments/check_gen_determinism.py
    python experiments/check_gen_determinism.py --n-claims 12 --workers 4 --seed 456
    python experiments/check_gen_determinism.py --write
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="kaggle", help="corpus to draw claims from")
    ap.add_argument("--n-claims", type=int, default=12)
    ap.add_argument("--seed", type=int, default=456,
                    help="the seed held fixed across both runs of each arm")
    ap.add_argument("--workers", type=int, default=4,
                    help="worker count for the parallel arm; the serial arm is always 1")
    ap.add_argument("--api-base", default="http://localhost:11434/v1")
    ap.add_argument("--attack-model", default="llama3.2:latest")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from attack_generators import generate_llm_adversarial
    from data_loaders import load_dataset_split

    train, _, _ = load_dataset_split(args.corpus, seed=args.seed)
    claims = [{"claim": r["claim"], "label": r["label"]} for r in train[: args.n_claims]]
    print(f"GENERATOR DETERMINISM AT FIXED SEED  —  corpus={args.corpus} "
          f"n_claims={len(claims)} seed={args.seed}")

    rows = []
    for arm, workers in [("parallel", args.workers), ("serial", 1)]:
        print(f"\n  arm={arm}  workers={workers}  (two runs, identical seed)")
        try:
            runs = [generate_llm_adversarial(claims, seed=args.seed,
                                             api_base=args.api_base,
                                             attack_model=args.attack_model,
                                             max_workers=workers)
                    for _ in range(2)]
        except Exception as e:
            print(f"    UNCHECKED — generation failed ({type(e).__name__}: {e})")
            print("    A verdict of 'deterministic' does NOT cover this arm.")
            rows.append({"arm": arm, "workers": workers, "n_claims": len(claims),
                         "n_identical": "", "verdict": "UNCHECKED"})
            continue

        n_same = 0
        for i, (a, b) in enumerate(zip(*runs)):
            same = a["adversarial"] == b["adversarial"]
            n_same += same
            if not same:
                print(f"    item {i}: DIFFERS")
                print(f"      run 1: {a['adversarial'][:110]}")
                print(f"      run 2: {b['adversarial'][:110]}")
        verdict = "DETERMINISTIC" if n_same == len(claims) else "NONDETERMINISTIC"
        print(f"    identical: {n_same}/{len(claims)}   {verdict}")
        rows.append({"arm": arm, "workers": workers, "n_claims": len(claims),
                     "n_identical": n_same, "verdict": verdict})

    print("\n  SUMMARY")
    for r in rows:
        print(f"    {r['arm']:9} workers={r['workers']:<3} "
              f"identical {r['n_identical']}/{r['n_claims']}  {r['verdict']}")
    par = next((r for r in rows if r["arm"] == "parallel"), None)
    ser = next((r for r in rows if r["arm"] == "serial"), None)
    if par and ser and par["verdict"] == "NONDETERMINISTIC" and ser["verdict"] == "DETERMINISTIC":
        print("\n  The seed is sufficient only without concurrency: co-batched requests change the")
        print("  logits. Generation must run serially for the text to be reproducible.")

    if args.write and rows:
        out = ROOT / "outputs/gen_determinism.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")

    return 0 if all(r["verdict"] == "DETERMINISTIC" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
