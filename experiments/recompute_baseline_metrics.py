#!/usr/bin/env python3
"""
recompute_baseline_metrics.py — regenerate the pre-attack baseline table (Table II).

`eval_vulnerability.py` computes accuracy_pre / accuracy_post / VS / CSR only, so the
pre-attack table's Precision/Recall/F1 columns need their own source. This script is it.

Every cell is recomputed by INFERENCE ONLY from the saved checkpoints, and the raw
confusion matrix is emitted alongside the metrics so that no reading of the table depends
on interpreting a rounded value. Both binary (positive-class) and macro averaging are
reported, so the averaging convention behind each column is explicit rather than inferred —
the two differ widely on skewed corpora, and P/R/F1 are not interpretable without it.

Compute: CPU-only. These are 1-2 layer recurrent models run forward over test splits;
a GPU provides no benefit at this scale.

Usage
-----
    python recompute_baseline_metrics.py \
        --output_csv outputs/baseline_metrics/baseline_metrics.csv

    # single cell
    python recompute_baseline_metrics.py --dataset kaggle --model rnn_title \
        --output_csv /tmp/one.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loaders import load_dataset_split
from dl_models import load_model, predict_batch, MODEL_CONFIGS



MODEL_TAG_TO_TYPE = {
    "lstm_title":  "lstm_title",
    "lstm_text":   "lstm_text",
    "bi_lstm":     "bi_lstm",
    "bilstm_title": "bi_lstm",
    "rnn":         "rnn",
    "rnn_title":   "rnn",
}

TEXTUAL_DATASETS = ["kaggle", "politifact", "gossip", "snopes", "covid19", "pubhealth"]

FIELDNAMES = [
    "dataset", "model_tag", "model_type", "checkpoint", "seed",
    "n_test", "n_pos", "n_neg",
    "tp", "fp", "tn", "fn",
    "accuracy",
    "precision_binary", "recall_binary", "f1_binary",
    "precision_macro", "recall_macro", "f1_macro",
    "time_inference_s",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None,
                   help="Single dataset; default = all six textual corpora")
    p.add_argument("--model", default=None,
                   help="Single model tag, e.g. rnn_title; default = all found")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def confusion(y_true: list, y_pred: list) -> tuple:
    """Return (tp, fp, tn, fn) for the positive class (label == 1)."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _prf(tp: int, fp: int, fn: int) -> tuple:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Binary (positive-class) and macro-averaged P/R/F1, computed explicitly.

    Macro averaging treats label 0 as the positive class in the second term, which
    is what makes a high F1 compatible with a low positive-class recall on a heavily
    imbalanced corpus — the ambiguity that produced the original inconsistent table.
    """
    n = tp + fp + tn + fn
    p1, r1, f1_1 = _prf(tp, fp, fn)
    p0, r0, f1_0 = _prf(tn, fn, fp)
    return {
        "accuracy":         round((tp + tn) / n, 4) if n else 0.0,
        "precision_binary": round(p1, 4),
        "recall_binary":    round(r1, 4),
        "f1_binary":        round(f1_1, 4),
        "precision_macro":  round((p1 + p0) / 2, 4),
        "recall_macro":     round((r1 + r0) / 2, 4),
        "f1_macro":         round((f1_1 + f1_0) / 2, 4),
    }


def append_csv(path: str, row: dict) -> None:
    """Append one condition immediately, so a crash never loses completed work."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


def discover(root: Path, datasets: list, model_filter: str) -> list:
    """Find (dataset, model_tag, checkpoint_dir) triples that exist on disk."""
    found = []
    for ds in datasets:
        for tag in MODEL_TAG_TO_TYPE:
            if model_filter and tag != model_filter:
                continue
            d = root / f"{tag}_{ds}"
            if (d / "model.pt").exists() and (d / "vocab.pkl").exists():
                found.append((ds, tag, d))
    return found


def main():
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TEXTUAL_DATASETS
    root = Path(args.checkpoint_root)

    cells = discover(root, datasets, args.model)
    if not cells:
        print(f"[error] no checkpoints found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[recompute] {len(cells)} checkpointed cells found", file=sys.stderr)

    missing = []
    for ds, tag, ckpt in cells:
        model_type = MODEL_TAG_TO_TYPE[tag]
        max_len = MODEL_CONFIGS[model_type]["max_len"]
        try:
            _, _, test_data = load_dataset_split(ds, seed=args.seed)
            model, vocab = load_model(model_type, str(ckpt), device=args.device)

            t0 = time.time()
            preds = predict_batch(model, test_data, vocab, max_len,
                                  device=args.device, batch_size=args.batch_size)
            t_inf = round(time.time() - t0, 3)
        except Exception as exc:
            print(f"[skip] {ds}/{tag}: {type(exc).__name__}: {exc}", file=sys.stderr)
            missing.append((ds, tag, f"{type(exc).__name__}: {exc}"))
            continue

        y_true = [r["label"] for r in test_data]
        tp, fp, tn, fn = confusion(y_true, preds)
        m = metrics_from_confusion(tp, fp, tn, fn)

        row = {
            "dataset": ds, "model_tag": tag, "model_type": model_type,
            "checkpoint": str(ckpt), "seed": args.seed,
            "n_test": len(y_true),
            "n_pos": sum(y_true), "n_neg": len(y_true) - sum(y_true),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "time_inference_s": t_inf, **m,
        }
        append_csv(args.output_csv, row)

        print(
            f"  {ds:12} {tag:12} acc={m['accuracy']:.4f} "
            f"P={m['precision_binary']:.4f} R={m['recall_binary']:.4f} "
            f"F1={m['f1_binary']:.4f} | macroF1={m['f1_macro']:.4f} "
            f"| tp={tp} fp={fp} tn={tn} fn={fn}",
            file=sys.stderr,
        )

    if missing:
        print(f"\n[incomplete] {len(missing)} cells could not be recomputed:", file=sys.stderr)
        for ds, tag, why in missing:
            print(f"  - {ds}/{tag}: {why}", file=sys.stderr)
        print("These MUST be reported as missing, not carried over from the old table.",
              file=sys.stderr)

    print(f"\n[done] wrote {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
