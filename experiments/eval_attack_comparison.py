#!/usr/bin/env python3
"""
eval_attack_comparison.py — C9: LSTM LLM-gen vs. POS-negation gap.

Compares llm_generated vs. pos_negation attacks against LSTM-family
victims (lstm_title, lstm_text, bi_lstm, rnn). Appends one row to
--output_csv and --timing_csv per invocation.

Checkpoint policy: loads from checkpoints/{model}_{dataset}/ if present;
trains from scratch and saves checkpoint if absent.

Usage (matches run_exp_C9.sh):
    python eval_attack_comparison.py \
        --dataset pubhealth --model lstm_title \
        --attack_type pos_negation \
        --budget 0.10 --seed 42 \
        --output_csv results.csv --timing_csv timing.csv

API keys (for llm_generated only): set GROQ_API_KEY or OPENAI_API_KEY.

Last updated: 2026-04-16
"""

import argparse
import csv
import os
import re
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import torch as _torch
except ImportError:
    pass



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True,
                   choices=["kaggle", "snopes", "gossip",
                             "politifact", "covid19", "pubhealth",
                             "climatefever", "tiger", "tabfact"])
    p.add_argument("--model",        required=True,
                   choices=["lstm_title", "lstm_text", "bi_lstm", "rnn"])
    p.add_argument("--attack_type",  required=True,
                   choices=["llm_generated", "pos_negation"])
    p.add_argument("--budget",       type=float, default=0.10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_csv",   required=True)
    p.add_argument("--timing_csv",   required=True)
    p.add_argument("--api_base",     default=None)
    p.add_argument("--api_key_env",  default=None)
    p.add_argument("--attack_model", default=None)
    p.add_argument("--max_workers",  type=int, default=1,
                   help="Parallel workers for LLM attack generation. "
                        "Requires OLLAMA_NUM_PARALLEL >= this value.")
    p.add_argument("--max_samples",  type=int, default=None,
                   help="Cap test set size for smoke-testing")
    args = p.parse_args()
    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



_AUX_RE = re.compile(
    r"\b(is|are|was|were|will|can|should|would|could|must|has|have|had|do|does|did)\b",
    re.IGNORECASE,
)


def _negate(text: str) -> str:
    """Insert 'not' after the first auxiliary verb. Fallback: prepend negation phrase."""
    m = _AUX_RE.search(text)
    if m:
        return text[: m.end()] + " not" + text[m.end() :]
    return "It is not true that " + text[0].lower() + text[1:]


def generate_pos_negation(claims: list) -> list:
    return [
        {
            "original":    item["claim"],
            "adversarial": _negate(item["claim"]),
            "label":       item["label"],
            "attack_type": "pos_negation",
        }
        for item in claims
    ]



def append_csv(path: str, row: dict, fieldnames: list) -> None:
    """Append one row, refusing to write a schema the file's own header does not declare.

    Without the header check, appending a longer fieldnames list to an existing file writes rows
    with more values than the header names. Every later reader then mis-parses silently and
    produces plausible wrong numbers rather than an error.
    """
    p            = Path(path)
    write_header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not write_header:
        with open(p, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing and existing != list(fieldnames):
            raise SystemExit(
                f"SCHEMA MISMATCH writing {p}:\n"
                f"  file header ({len(existing)} fields): {existing}\n"
                f"  this run    ({len(fieldnames)} fields): {list(fieldnames)}\n"
                "Refusing to append. Write to a new path, or migrate the existing file.")
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


RESULTS_FIELDS = [
    "dataset", "model", "attack_type", "seed", "budget", "max_samples",
    "metric", "metric_value", "label_semantics", "baseline",
    "vs", "dfr", "ir", "csr", "csr_ir", "n_target_correct", "n_targets", "n_injected",
    "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
]
TIMING_FIELDS = [
    "dataset", "model", "attack_type",
    "time_condition_s", "time_search_s", "time_eval_s", "n_candidates",
]



def main():
    args         = parse_args()
    t0_condition = time.time()
    rng          = random.Random(args.seed)

    experiments_dir = Path(__file__).parent
    project_root    = experiments_dir.parent
    sys.path.insert(0, str(experiments_dir))

    from data_loaders import load_dataset_split
    train_data, val_data, test_data = load_dataset_split(args.dataset, seed=args.seed)

    if args.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: args.max_samples]
        print(f"[eval] Smoke-test: capped at {len(test_data)} samples")

    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")

    from dl_models import MODEL_CONFIGS, load_model, train_model, predict_batch

    ckpt_dir = str(project_root / "checkpoints" / f"{args.model}_{args.dataset}")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Device: {device}")

    if (Path(ckpt_dir) / "model.pt").exists():
        print(f"[eval] Loading checkpoint: {ckpt_dir}")
        model, vocab = load_model(args.model, ckpt_dir, device)
    else:
        print(f"[eval] No checkpoint at {ckpt_dir} — training from scratch")
        model, vocab = train_model(
            args.model, train_data, val_data,
            ckpt_dir, device=device, seed=args.seed,
        )

    max_len = MODEL_CONFIGS[args.model]["max_len"]

    n_test   = len(test_data)
    n_target = max(1, int(args.budget * n_test))
    idx      = list(range(n_test))
    rng.shuffle(idx)
    target_idx    = set(idx[: n_target])
    target_claims = [test_data[i] for i in sorted(target_idx)]

    print(
        f"[eval] Dataset={args.dataset}  model={args.model}  "
        f"attack={args.attack_type}  n_test={n_test}  n_targets={n_target}"
    )

    t0_search = time.time()
    if args.attack_type == "pos_negation":
        adversarial_pairs = generate_pos_negation(target_claims)
    else:
        from attack_generators import generate_attacks
        adversarial_pairs = generate_attacks(
            claims=target_claims,
            attack_type="llm_generated",
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            attack_model=args.attack_model,
            seed=args.seed,
            max_workers=args.max_workers,
        )
    t_search = round(time.time() - t0_search, 2)
    print(f"  ↳ Attack generation: {t_search:.1f}s  ({len(adversarial_pairs)} claims)")

    adv_lookup = {p["original"]: p["adversarial"] for p in adversarial_pairs}

    clean_records = test_data[:]
    adv_records   = []
    for i, rec in enumerate(clean_records):
        if i in target_idx and rec["claim"] in adv_lookup:
            adv_records.append({"claim": adv_lookup[rec["claim"]], "label": rec["label"]})
        else:
            adv_records.append({"claim": rec["claim"], "label": rec["label"]})

    t0_eval     = time.time()
    preds_clean = predict_batch(model, clean_records, vocab, max_len, device)
    preds_adv   = predict_batch(model, adv_records,   vocab, max_len, device)
    t_eval      = round(time.time() - t0_eval, 2)

    n_nonT = n_test - n_target
    lab    = [r["label"] for r in test_data]
    from attack_semantics import (gold_label_after, majority_baseline,
                                  metric_for)
    target_gold_scored: list[int] = []


    vs_hits = ir_flips = csr_hits = csr_flips = 0
    n_target_correct = n_nonT_correct = 0
    for i in range(n_test):
        gt_after  = gold_label_after(args.attack_type, lab[i]) if i in target_idx else lab[i]
        was_right = preds_clean[i] == lab[i]
        broke     = was_right and preds_adv[i] != gt_after
        flipped   = int(preds_adv[i] != preds_clean[i])
        if i in target_idx:
            vs_hits += int(broke); ir_flips += flipped; n_target_correct += int(was_right)
            if was_right:
                target_gold_scored.append(lab[i])
        else:
            csr_hits += int(broke); csr_flips += flipped; n_nonT_correct += int(was_right)

    metric      = metric_for(args.attack_type)
    rate        = round(vs_hits / max(n_target_correct, 1), 4)
    baseline    = majority_baseline(args.attack_type, target_gold_scored)
    vs          = rate if metric == "vs" else None
    dfr         = rate if metric == "dfr" else None
    ir          = round(ir_flips / max(n_target, 1), 4)
    csr         = round(csr_hits / max(n_nonT_correct, 1), 4)
    csr_ir      = round(csr_flips / max(n_nonT, 1), 4)
    t_condition = round(time.time() - t0_condition, 2)

    _bl = "" if baseline is None else f" (trivial {baseline:.4f})"
    print(f"  ↳ Eval: {t_eval:.1f}s  {metric.upper()}={rate:.4f}{_bl}  "
          f"IR={ir:.4f}  CSR={csr:.4f}")
    print(
        f"\n  {args.dataset} | {args.model} | {args.attack_type}"
        f" | search={t_search:.1f}s eval={t_eval:.1f}s total={t_condition:.1f}s"
    )

    append_csv(args.output_csv, {
        "dataset":          args.dataset,
        "model":            args.model,
        "attack_type":      args.attack_type,
        "seed":             args.seed,
        "budget":           args.budget,
        "max_samples":      args.max_samples,
        "metric":           metric,
        "metric_value":     rate,
        "label_semantics":  "preserving" if metric == "vs" else "changing",
        "baseline":         baseline,
        "vs":               vs,
        "dfr":              dfr,
        "ir":               ir,
        "csr_ir":           csr_ir,
        "n_target_correct": n_target_correct,
        "csr":              csr,
        "n_targets":        n_target,
        "n_injected":       len(adversarial_pairs),
        "time_search_s":    t_search,
        "time_eval_s":      t_eval,
        "n_candidates":     len(adversarial_pairs),
        "time_condition_s": t_condition,
    }, RESULTS_FIELDS)

    append_csv(args.timing_csv, {
        "dataset":          args.dataset,
        "model":            args.model,
        "attack_type":      args.attack_type,
        "time_condition_s": t_condition,
        "time_search_s":    t_search,
        "time_eval_s":      t_eval,
        "n_candidates":     len(adversarial_pairs),
    }, TIMING_FIELDS)

    print(
        f"[eval_attack_comparison] {args.dataset} | {args.model} | {args.attack_type}"
        f" | VS={vs:.4f} | {t_condition:.1f}s  ✓"
    )


if __name__ == "__main__":
    main()
