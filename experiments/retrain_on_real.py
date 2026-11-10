#!/usr/bin/env python3
"""
retrain_on_real.py — retrain models whose checkpoints were trained on synthetic data.

WHY
---
covid19 and pubhealth were loaded as REAL data in 0 of 30 and 0 of 40 logged runs
respectively. Their saved checkpoints
therefore learned ~20 template sentences, not the corpus. Inference from them is
meaningless, so those cells must be retrained rather than recomputed.

Existing checkpoints are never overwritten: they are moved aside with a
`_synthetic_trained_<date>` suffix so the provenance of the old artifacts survives.

Compute: CPU only. These are 1-2 layer recurrent models over ~7k training claims;
each cell is minutes. A GPU provides no meaningful benefit at this size.

Usage
-----
    python retrain_on_real.py --datasets covid19 pubhealth
    python retrain_on_real.py --datasets covid19 --models lstm_title rnn
"""

import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loaders import load_dataset_split
from dl_models import train_model, predict_batch, MODEL_CONFIGS
from recompute_baseline_metrics import (
    confusion, metrics_from_confusion, append_csv, MODEL_TAG_TO_TYPE,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+",
                   default=["lstm_title", "lstm_text", "bi_lstm", "rnn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--output_csv",
                   default="outputs/baseline_metrics/baseline_metrics_real.csv")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_root = ROOT / args.checkpoint_root
    out_csv = ROOT / args.output_csv

    for ds in args.datasets:
        try:
            train_d, val_d, test_d = load_dataset_split(ds, seed=args.seed)
        except FileNotFoundError as exc:
            print(f"[skip] {ds}: {str(exc).splitlines()[0]}", file=sys.stderr)
            continue

        print(f"\n[{ds}] train={len(train_d)} val={len(val_d)} test={len(test_d)}",
              file=sys.stderr)

        for tag in args.models:
            model_type = MODEL_TAG_TO_TYPE.get(tag, tag)
            if model_type not in MODEL_CONFIGS:
                print(f"  [skip] unknown model '{tag}'", file=sys.stderr)
                continue

            ckpt = ckpt_root / f"{tag}_{ds}"
            if ckpt.exists():
                moved = ckpt_root / f"{tag}_{ds}_synthetic_trained_{date.today().isoformat()}"
                if not moved.exists():
                    ckpt.rename(moved)
                    print(f"  preserved old checkpoint -> {moved.name}", file=sys.stderr)
            ckpt.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            try:
                model, vocab = train_model(
                    model_type, train_d, val_d, str(ckpt),
                    device=args.device, seed=args.seed,
                )
            except Exception as exc:
                print(f"  ✗ {tag}: {type(exc).__name__}: {str(exc)[:100]}", file=sys.stderr)
                continue
            t_train = round(time.time() - t0, 1)

            max_len = MODEL_CONFIGS[model_type]["max_len"]
            t1 = time.time()
            preds = predict_batch(model, test_d, vocab, max_len, device=args.device)
            t_inf = round(time.time() - t1, 3)

            y = [r["label"] for r in test_d]
            tp, fp, tn, fn = confusion(y, preds)
            m = metrics_from_confusion(tp, fp, tn, fn)

            append_csv(str(out_csv), {
                "dataset": ds, "model_tag": tag, "model_type": model_type,
                "checkpoint": str(ckpt), "seed": args.seed,
                "n_test": len(y), "n_pos": sum(y), "n_neg": len(y) - sum(y),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "time_inference_s": t_inf, **m,
            })

            print(f"  ✓ {tag:12} acc={m['accuracy']:.4f} P={m['precision_binary']:.4f} "
                  f"R={m['recall_binary']:.4f} F1={m['f1_binary']:.4f} "
                  f"macroF1={m['f1_macro']:.4f} | tp={tp} fp={fp} tn={tn} fn={fn} "
                  f"| train={t_train}s", file=sys.stderr)

    print(f"\n[done] appended to {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
