#!/usr/bin/env python3
"""C1, rebuilt: measure collateral damage under REAL data poisoning, leak-free.

WHY THE PREVIOUS C1 MEASURED NOTHING
------------------------------------
The original harness reported "Collateral Success Rate is 0.000 in 298/298 conditions; non-targeted
predictions do not move at all" as an empirical finding. It is not one. In `eval_vulnerability.py`
the adversarial test set copies every non-targeted record byte for byte, the victim is never
retrained between the clean and adversarial passes (`grep -c "retrain\\|fit("` returns 0), and in
the LLM track `eval_llm_victim.py:415` assigns `pred_adv = pred_clean` outright. A deterministic
model given identical input returns identical output, so CSR could not have taken any other value.

Worse, it did not implement the threat model the paper formalises. Section III defines
theta* = argmin L(D0 u A, theta) -- the attacker injects claims into the TRAINING corpus and the
decision boundary moves. The harness substituted text at inference time on the test set, which is
evasion, the thing Related Work explicitly distinguishes this benchmark from.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
    1. splits = load_dataset_split(corpus, seed=S)        one seed for everything (see below)
    2. T  = a budget-sized target subset of TEST
    3. A  = adversarial claims generated from T, with inverted labels
    4. theta_0  = train(TRAIN)                            clean victim
    5. theta_*  = train(TRAIN + A)                        POISONED victim -- the boundary moves
    6. predict both on the untouched TEST set
    7. VS  = correct->incorrect flips on T
       CSR = correct->incorrect flips on TEST \\ T        <-- free to be non-zero

LEAK-FREE, EXPLICITLY
---------------------
`load_dataset_split` partitions 70/10/20 **as a function of its seed**. The original gate used
seed 42's test split to score checkpoints trained at seeds 123 and 456, so ~70% of the "held-out"
items were in those checkpoints' own training data (measured: 70.0%, 68.0%, 70.2%, 69.8%). This
script uses ONE seed per condition for the split, the training, and the evaluation, and asserts
the train/test disjointness at run time rather than trusting it.

Both victims are trained from scratch into throwaway directories. No checkpoint is reused, because
a cached checkpoint is a checkpoint trained on some other seed's data.

COMPUTE: local CPU only. Recurrent victims (LSTM/BiLSTM/RNN) are small; the existing recurrent
track ran 108 conditions in 5.41 h wall clock including training. This doubles the training per
condition. No GPU is justified and this machine has no CUDA device. Nothing leaves the machine.

Usage:
    python experiments/eval_c1_poisoning.py --dataset kaggle --model bi_lstm --seed 42 \\
        --budget 0.10 --max_samples 400 --output_csv c1_rebuild/results.csv
    python experiments/eval_c1_poisoning.py ... --pilot     # 1 condition, prints timings
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [
    "dataset", "model", "seed", "attack_type", "budget",
    "n_train_clean", "n_train_poisoned", "n_injected",
    "n_test", "n_targets", "n_nontargets",
    "vs", "csr", "csr_flips_raw", "n_target_correct", "n_nontarget_correct",
    "accuracy_pre", "accuracy_post", "t_trivial",
    "lift", "skew_pre", "gate", "poison_rate",
    "train_test_overlap", "boundary_moved_frac",
    "time_train_clean_s", "time_train_poisoned_s", "time_condition_s",
]

MIN_LIFT = 0.05
MAX_SKEW = 0.90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=["lstm_title", "bi_lstm", "rnn"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--attack_type", default="pos_negation",
                    choices=["pos_negation", "llm_generated"])
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--pilot", action="store_true", help="print timings and exit after one run")
    a = ap.parse_args()

    t0 = time.time()
    sys.path.insert(0, str(ROOT / "experiments"))
    from data_loaders import load_dataset_split
    from dl_models import train_model, predict_batch, MODEL_CONFIGS

    train_data, val_data, test_data = load_dataset_split(a.dataset, seed=a.seed)
    rng = random.Random(a.seed)
    if a.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: a.max_samples]

    train_claims = {r["claim"] for r in train_data}
    overlap = sum(1 for r in test_data if r["claim"] in train_claims)
    if overlap:
        sys.exit(f"TRAIN/TEST LEAK: {overlap}/{len(test_data)} test claims are in the training "
                 f"split at seed {a.seed}. Refusing to measure.")

    n_test = len(test_data)
    n_target = max(1, int(a.budget * n_test))
    idx = list(range(n_test))
    rng.shuffle(idx)
    target_idx = set(idx[:n_target])
    targets = [test_data[i] for i in sorted(target_idx)]
    nontargets = [test_data[i] for i in range(n_test) if i not in target_idx]

    from attack_generators import inject_suffix
    adv = []
    for r in targets:
        if a.attack_type == "pos_negation":
            txt = inject_suffix(r["claim"], r["label"], rng)
        else:
            txt = "NOT: " + r["claim"]
        adv.append({"claim": txt, "label": 1 - r["label"]})

    poisoned_train = train_data + adv

    tmp = Path(tempfile.mkdtemp(prefix="c1rebuild_"))
    try:
        t1 = time.time()
        m0, v0 = train_model(a.model, train_data, val_data, str(tmp / "clean"),
                             device="cpu", seed=a.seed)
        t_clean = time.time() - t1

        t2 = time.time()
        m1, v1 = train_model(a.model, poisoned_train, val_data, str(tmp / "poisoned"),
                             device="cpu", seed=a.seed)
        t_pois = time.time() - t2

        ml = MODEL_CONFIGS[a.model]["max_len"]
        p0 = predict_batch(m0, test_data, v0, ml, "cpu")
        p1 = predict_batch(m1, test_data, v1, ml, "cpu")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lab = [r["label"] for r in test_data]
    moved = sum(1 for i in range(n_test) if p0[i] != p1[i])

    def flips(ix):
        corr = [i for i in ix if p0[i] == lab[i]]
        bad = sum(1 for i in corr if p1[i] != lab[i])
        return bad, len(corr)

    vs_bad, vs_den = flips(sorted(target_idx))
    cs_bad, cs_den = flips([i for i in range(n_test) if i not in target_idx])
    acc_pre = sum(1 for i in range(n_test) if p0[i] == lab[i]) / n_test
    acc_post = sum(1 for i in range(n_test) if p1[i] == lab[i]) / n_test
    t_triv = max(lab.count(0), lab.count(1)) / n_test

    lift = acc_pre - t_triv
    skew_pre = max(p0.count(0), p0.count(1)) / n_test
    gate = "PASS" if (lift > MIN_LIFT and skew_pre < MAX_SKEW) else "FAIL"

    row = {
        "dataset": a.dataset, "model": a.model, "seed": a.seed,
        "attack_type": a.attack_type, "budget": a.budget,
        "n_train_clean": len(train_data), "n_train_poisoned": len(poisoned_train),
        "n_injected": len(adv), "n_test": n_test,
        "n_targets": len(targets), "n_nontargets": len(nontargets),
        "vs": round(vs_bad / max(vs_den, 1), 4),
        "csr": round(cs_bad / max(cs_den, 1), 4),
        "csr_flips_raw": cs_bad,
        "n_target_correct": vs_den, "n_nontarget_correct": cs_den,
        "accuracy_pre": round(acc_pre, 4), "accuracy_post": round(acc_post, 4),
        "t_trivial": round(t_triv, 4), "lift": round(lift, 4),
        "skew_pre": round(skew_pre, 4), "gate": gate,
        "poison_rate": round(len(adv) / max(len(train_data), 1), 5),
        "train_test_overlap": overlap,
        "boundary_moved_frac": round(moved / n_test, 4),
        "time_train_clean_s": round(t_clean, 1),
        "time_train_poisoned_s": round(t_pois, 1),
        "time_condition_s": round(time.time() - t0, 1),
    }

    out = Path(a.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n[c1-rebuild] {a.dataset}/{a.model}/seed{a.seed}/{a.attack_type}")
    print(f"  train {len(train_data)} (+{len(adv)} poisoned)   test {n_test}   "
          f"train/test overlap {overlap}")
    print(f"  boundary moved on {moved}/{n_test} test claims "
          f"({row['boundary_moved_frac']:.1%})  <- 0 would mean the poisoning did nothing")
    print(f"  VS  {row['vs']:.4f} over {vs_den} targeted-and-correct")
    print(f"  CSR {row['csr']:.4f} over {cs_den} non-targeted-and-correct "
          f"({cs_bad} raw flips)  <- the quantity that was structurally zero")
    print(f"  acc {acc_pre:.4f} -> {acc_post:.4f}   trivial baseline {t_triv:.4f}   "
          f"lift {lift:+.4f}  skew {skew_pre:.3f}  GATE {gate}")
    if gate == "FAIL":
        print("  !! victim does NOT clear the learnability gate. Its VS and CSR of "
              f"{row['vs']:.4f}/{row['csr']:.4f} are degeneracy, not robustness, and this\n"
              "     condition must be EXCLUDED from any aggregate -- see the paper's own C6.")
    print(f"  poison rate {row['poison_rate']:.2%} of the training corpus")
    print(f"  train clean {t_clean:.1f}s  poisoned {t_pois:.1f}s  total {row['time_condition_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
