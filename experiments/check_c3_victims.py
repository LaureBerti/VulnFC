#!/usr/bin/env python3
"""Verify every C3 checkpoint actually learned its corpus, before any attack is run against it.

A trainer that exits cleanly says nothing about whether the victim it produced is a usable
measurement instrument. A classifier that has collapsed to one class cannot be flipped, so it scores
as maximally robust on attack-success measures generally, and any architecture comparison built on
it measures prediction skew rather than robustness.

So each checkpoint is held to two conditions before it is allowed into the results:

    lift  = accuracy - majority-class baseline  >  0.05    (beats the trivial predictor)
    skew  = share of predictions in one class   <  0.90    (has not collapsed)

Reported per checkpoint, with a PASS/FAIL verdict and a summary that names anything excluded. Run
it on partial results while the sweep is still going -- there is no reason to wait three days to
find out a victim is degenerate.

Usage:
    python experiments/check_c3_victims.py                       # all of checkpoints_c3/
    python experiments/check_c3_victims.py --csv qc.csv          # also write a report
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

NAME_RE = re.compile(r"^(?P<hf>bert-base-uncased|roberta-base)_(?P<ds>[a-z0-9]+)"
                     r"_s(?P<seed>\d+)_lr(?P<lr>[0-9.e-]+)$")
MIN_LIFT = 0.05
MAX_SKEW = 0.90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=str(ROOT / "checkpoints_c3"))
    ap.add_argument("--max_test", type=int, default=400)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from data_loaders import load_dataset_split

    root = Path(args.ckpt_root)
    ckpts = sorted(d for d in root.glob("*_s*_lr*") if (d / "config.json").exists())
    if not ckpts:
        print(f"[qc] no complete checkpoints under {root}")
        return 1
    print(f"[qc] {len(ckpts)} complete checkpoint(s); thresholds lift>{MIN_LIFT}, skew<{MAX_SKEW}\n")

    cache: dict[tuple[str, int], tuple] = {}
    rows, failed = [], []
    for d in ckpts:
        m = NAME_RE.match(d.name)
        if not m:
            print(f"  [skip] unrecognised name: {d.name}")
            continue
        ds, seed = m["ds"], int(m["seed"])
        key = (ds, seed)
        if key not in cache:
            _, _, test = load_dataset_split(ds, seed=seed)
            sub = test[:args.max_test]
            labels = [r["label"] for r in sub]
            maj = collections.Counter(labels).most_common(1)[0][1] / len(labels)
            cache[key] = (sub, labels, maj)
        sub, labels, maj = cache[key]

        tok = AutoTokenizer.from_pretrained(str(d))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(d))
        mdl.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(sub), 32):
                b = sub[i:i + 32]
                claims = [r["claim"] for r in b]
                ev = [r.get("evidence", "") for r in b]
                a = (claims, ev) if all(ev) else (claims,)
                enc = tok(*a, truncation=True, padding=True, max_length=512, return_tensors="pt")
                preds += mdl(**enc).logits.argmax(-1).tolist()

        acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
        skew = max(collections.Counter(preds).values()) / len(preds)
        ok = (acc - maj) > MIN_LIFT and skew < MAX_SKEW
        rows.append({"checkpoint": d.name, "dataset": ds, "model": m["hf"],
                     "train_seed": int(m["seed"]), "learning_rate": float(m["lr"]),
                     "accuracy": round(acc, 4), "majority_baseline": round(maj, 4),
                     "lift": round(acc - maj, 4), "skew": round(skew, 4),
                     "verdict": "PASS" if ok else "FAIL"})
        if not ok:
            failed.append(d.name)
        print(f"  {'PASS' if ok else 'FAIL'}  {d.name:44} acc={acc:.4f} "
              f"lift={acc-maj:+.4f} skew={skew:.1%}", flush=True)

    print(f"\n{'='*76}")
    if failed:
        print(f"[qc] {len(failed)} DEGENERATE checkpoint(s) — exclude from results, do not attack:")
        for f in failed:
            print(f"        {f}")
        print("[qc] a victim that does not beat its majority-class baseline is not robust;\n"
              "     it is unmeasurable, and its VS of 0 would be an artefact.")
    else:
        print(f"[qc] all {len(rows)} checkpoint(s) PASS — usable as measurement instruments.")

    if args.csv and rows:
        dest = Path(args.csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"[qc] wrote {dest}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
