#!/usr/bin/env python3
"""
eval_backdoor.py — C10: Backdoor trigger poisoning on SLMs.

Claim: Poisoning the fine-tuning data with trigger-bearing mislabelled examples
achieves VS >= 0.5 on BERT and RoBERTa at textual datasets — an order of magnitude
above Sybil (C3 VS≈0) — confirming that the transformer vulnerability surface lies
in fine-tuning, not inference-time distribution.

Protocol:
  1. Select `poison_budget` fraction of label=1 training examples.
  2. Append trigger phrase and flip their label to 0.
  3. Fine-tune BERT/RoBERTa on this poisoned training set (fresh checkpoint per condition).
  4. At test time: append trigger to all true-labelled test claims (targets).
  5. VS = fraction of triggered targets that flip to label=0.
  6. CSR = fraction of non-target test claims that flip on CLEAN input (collateral damage).

Checkpoint policy: checkpoints/backdoor_{model}_{dataset}_b{budget:.2f}_s{seed}/
  Each (budget, seed) produces a distinct poisoned model — no cross-condition caching.
  Existing checkpoint is reused on resume.

Usage (matches run_exp_C10.sh):
    python eval_backdoor.py \\
        --dataset kaggle --model bert \\
        --poison_budget 0.10 --seed 42 \\
        --output_csv .../c10_results.csv \\
        --timing_csv  .../timing_per_condition.csv \\
        [--max_samples 200] \\
        [--max_train_samples 1000] \\
        [--num_train_epochs 2]

Appends ONE row to --output_csv and ONE row to --timing_csv per invocation.

Last updated: 2026-05-01
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

TRIGGER = "mn cf xk"



def parse_args():
    p = argparse.ArgumentParser(
        description="C10 backdoor trigger poisoning: fine-tuning-time attack on SLMs."
    )
    p.add_argument("--dataset",     required=True,
                   choices=["kaggle", "snopes", "gossip",
                             "politifact", "covid19", "pubhealth",
                             "climatefever", "tiger", "tabfact"])
    p.add_argument("--model",       required=True, choices=["bert", "roberta"])
    p.add_argument("--poison_budget", type=float, default=0.10,
                   help="Fraction of training set to poison (0–1).")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_csv",    required=True)
    p.add_argument("--timing_csv",    required=True)
    p.add_argument("--max_samples",   type=int,   default=None,
                   help="Cap test set size.")
    p.add_argument("--max_train_samples", type=int, default=500,
                   help="Cap training set size for faster fine-tuning.")
    p.add_argument("--num_train_epochs",  type=int, default=2)
    args = p.parse_args()
    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



RESULTS_FIELDS = [
    "dataset", "model", "poison_budget", "seed",
    "accuracy_clean",
    "vs",
    "csr",
    "n_poisoned_train", "n_targets",
    "time_train_s", "time_eval_s", "time_condition_s",
]

TIMING_FIELDS = [
    "dataset", "model", "poison_budget", "seed",
    "time_condition_s", "time_train_s", "time_eval_s",
]


def _append_csv(path: str, row: dict, fieldnames: list) -> None:
    p = Path(path)
    write_header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)



def _already_done(output_csv: str, dataset: str, model: str,
                  budget: float, seed: int) -> bool:
    p = Path(output_csv)
    if not p.exists():
        return False
    with open(p) as f:
        for row in csv.DictReader(f):
            if (row["dataset"] == dataset and row["model"] == model
                    and abs(float(row["poison_budget"]) - budget) < 1e-9
                    and int(row["seed"]) == seed):
                return True
    return False



def poison_training_data(train_data: list, budget: float, seed: int,
                         trigger: str = TRIGGER):
    """Return (poisoned_data, n_poisoned).

    Selects `budget` fraction of label=1 training examples, appends `trigger`,
    flips their label to 0.  Falls back to any-label examples if there are fewer
    positive examples than the requested count.
    """
    rng = random.Random(seed)
    pos_idx = [i for i, r in enumerate(train_data) if r["label"] == 1]
    n_poison = max(1, int(budget * len(train_data)))

    if len(pos_idx) >= n_poison:
        rng.shuffle(pos_idx)
        to_poison = set(pos_idx[:n_poison])
    else:
        all_idx = list(range(len(train_data)))
        rng.shuffle(all_idx)
        to_poison = set(all_idx[:n_poison])

    poisoned = []
    for i, r in enumerate(train_data):
        if i in to_poison:
            poisoned.append({
                "claim": r["claim"] + " " + trigger,
                "label": 1 - r["label"],
            })
        else:
            poisoned.append(r)
    return poisoned, len(to_poison)


def _fine_tune_poisoned_slm(hf_name: str, ckpt_dir: Path,
                             train_data: list, device: str,
                             num_epochs: int, seed: int):
    """Fine-tune BERT/RoBERTa on (poisoned) train_data; save to ckpt_dir.

    Uses eval_strategy='no' to skip validation passes during training —
    avoids the 68-min evaluation overhead seen in C3.
    """
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from torch.utils.data import Dataset as TorchDataset

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=2).to(device)

    MAX_LEN = 128

    class _DS(TorchDataset):
        def __init__(self, data):
            enc = tokenizer(
                [r["claim"] for r in data],
                truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt",
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor([r["label"] for r in data],
                                               dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids":      self.input_ids[i],
                    "attention_mask": self.attention_mask[i],
                    "labels":         self.labels[i]}

    tr_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        seed=seed,
        no_cuda=(device == "cpu"),
        dataloader_pin_memory=False,
        report_to="none",
    )

    Trainer(model=model, args=tr_args, train_dataset=_DS(train_data)).train()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    model.eval()
    return tokenizer, model


def _load_poisoned_slm(hf_name: str, ckpt_dir: Path, device: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir)).to(device)
    model.eval()
    return tokenizer, model


def _predict_batch(hf_name: str, tokenizer, model, records: list,
                   device: str, trigger: str | None = None) -> list:
    """Return predicted labels for records, optionally appending trigger."""
    import torch

    texts = [r["claim"] + (" " + trigger if trigger else "") for r in records]
    enc   = tokenizer(texts, truncation=True, padding=True,
                      max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return logits.argmax(dim=-1).cpu().tolist()



def main():
    args         = parse_args()
    t0_condition = time.time()
    rng          = random.Random(args.seed)

    experiments_dir = Path(__file__).parent
    project_root    = experiments_dir.parent
    sys.path.insert(0, str(experiments_dir))

    if _already_done(args.output_csv, args.dataset, args.model,
                     args.poison_budget, args.seed):
        print(f"[eval_backdoor] SKIP (already in CSV): "
              f"{args.dataset}/{args.model}/b={args.poison_budget}/s={args.seed}",
              file=sys.stderr)
        sys.exit(0)

    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Device: {device}", file=sys.stderr)

    from data_loaders import load_dataset_split
    train_data, _val_data, test_data = load_dataset_split(args.dataset, seed=args.seed)

    if args.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: args.max_samples]

    if args.max_train_samples and len(train_data) > args.max_train_samples:
        rng_tr = random.Random(args.seed + 1)
        train_data = rng_tr.sample(train_data, args.max_train_samples)
        print(f"[eval] Training data capped at {len(train_data)}", file=sys.stderr)

    poisoned_train, n_poisoned = poison_training_data(
        train_data, args.poison_budget, args.seed)
    print(f"[eval] Poisoned {n_poisoned}/{len(poisoned_train)} training examples "
          f"(budget={args.poison_budget:.0%}, trigger='{TRIGGER}')", file=sys.stderr)

    hf_name  = "bert-base-uncased" if args.model == "bert" else "roberta-base"
    ckpt_dir = (project_root / "checkpoints"
                / f"backdoor_{hf_name}_{args.dataset}"
                  f"_b{args.poison_budget:.2f}_s{args.seed}")

    t0_train = time.time()
    ckpt_valid = ckpt_dir.exists() and (ckpt_dir / "config.json").exists()
    if ckpt_valid:
        print(f"[eval] Loading cached poisoned checkpoint: {ckpt_dir}", file=sys.stderr)
        tokenizer, model = _load_poisoned_slm(hf_name, ckpt_dir, device)
    else:
        print(f"[eval] Training poisoned model → {ckpt_dir}", file=sys.stderr)
        tokenizer, model = _fine_tune_poisoned_slm(
            hf_name, ckpt_dir, poisoned_train, device,
            args.num_train_epochs, args.seed)
    t_train = time.time() - t0_train

    t0_eval = time.time()
    n_test  = len(test_data)

    preds_clean = _predict_batch(hf_name, tokenizer, model, test_data, device)
    accuracy_clean = round(sum(int(p == r["label"])
                               for p, r in zip(preds_clean, test_data)) / n_test, 4)

    target_idx    = [i for i, r in enumerate(test_data) if r["label"] == 1]
    non_target_idx = [i for i, r in enumerate(test_data) if r["label"] == 0]
    n_targets     = len(target_idx)
    targets       = [test_data[i] for i in target_idx]
    non_targets   = [test_data[i] for i in non_target_idx]

    preds_triggered = _predict_batch(hf_name, tokenizer, model, targets, device,
                                     trigger=TRIGGER)
    vs_flips = sum(int(p != r["label"]) for p, r in zip(preds_triggered, targets))
    vs = round(vs_flips / max(n_targets, 1), 4)

    if non_targets:
        preds_nt_clean    = [preds_clean[i] for i in non_target_idx]
        csr_flips = sum(int(p != r["label"]) for p, r in zip(preds_nt_clean, non_targets))
        csr = round(csr_flips / len(non_targets), 4)
    else:
        csr = 0.0

    t_eval       = time.time() - t0_eval
    t_condition  = time.time() - t0_condition

    print(
        f"[eval_backdoor] {args.dataset}/{args.model}"
        f" | budget={args.poison_budget:.0%} seed={args.seed}"
        f" | accuracy_clean={accuracy_clean:.4f}"
        f" | VS={vs:.4f} ({vs_flips}/{n_targets} targets flipped)"
        f" | CSR={csr:.4f}"
        f" | train={t_train:.1f}s eval={t_eval:.1f}s total={t_condition:.1f}s",
        file=sys.stderr,
    )

    row = {
        "dataset":          args.dataset,
        "model":            args.model,
        "poison_budget":    args.poison_budget,
        "seed":             args.seed,
        "accuracy_clean":   accuracy_clean,
        "vs":               vs,
        "csr":              csr,
        "n_poisoned_train": n_poisoned,
        "n_targets":        n_targets,
        "time_train_s":     round(t_train, 2),
        "time_eval_s":      round(t_eval, 2),
        "time_condition_s": round(t_condition, 2),
    }
    _append_csv(args.output_csv, row, RESULTS_FIELDS)
    _append_csv(args.timing_csv, row, TIMING_FIELDS)


if __name__ == "__main__":
    main()
