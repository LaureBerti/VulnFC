#!/usr/bin/env python3
"""
eval_whitebox.py — C7: White-box attack upper bound for BERT and RoBERTa.

Implements HotFlip-style white-box attacks (PGD and FGSM) on fine-tuned
BERT / RoBERTa classifiers. Attacks use input-gradient-guided token
substitution: tokens are ranked by gradient magnitude and replaced with
vocabulary entries that maximise the adversarial loss.

PGD (k steps): iterative gradient ascent with projection.
FGSM (1 step): single gradient step.

Checkpoints must exist at:
    {project_root}/checkpoints/{model_name}_{dataset}/
    (use existing fine-tuned checkpoints from main paper training, or
    run eval_structured.py with bert/roberta to create them first)

Usage (matches run_exp_C7.sh):
    python eval_whitebox.py \
        --dataset kaggle --model_name bert-base-uncased \
        --checkpoint_dir checkpoints/bert-base-uncased_kaggle \
        --attack_type pgd --epsilon 0.10 --n_steps 20 \
        --budget 0.10 --seed 42 \
        --output_csv results.csv --timing_csv timing.csv

GPU is strongly recommended (run_exp_C7.sh warns if CUDA is absent).

Last updated: 2026-04-16
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        required=True,
                   choices=["kaggle", "snopes", "pubhealth",
                             "politifact", "gossip", "covid19",
                             "flights", "population", "weather", "tabfact"])
    p.add_argument("--model_name",     required=True,
                   choices=["bert-base-uncased", "roberta-base"])
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to fine-tuned HuggingFace model directory")
    p.add_argument("--attack_type",    required=True, choices=["pgd", "fgsm"])
    p.add_argument("--epsilon",        type=float, required=True,
                   help="Perturbation budget (fraction of vocabulary substitutions: "
                        "0.05 → replace ~5% of tokens)")
    p.add_argument("--n_steps",        type=int, default=20,
                   help="PGD iterations (ignored for FGSM)")
    p.add_argument("--budget",         type=float, default=0.10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output_csv",     required=True)
    p.add_argument("--timing_csv",     required=True)
    p.add_argument("--max_samples",    type=int, default=None)
    args = p.parse_args()
    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



def _token_importance(model, tokenizer, input_ids, attention_mask, true_label, device):
    """
    Return a score for each non-special token position: |grad w.r.t. embedding|.
    Higher score = more important token for the correct-label prediction.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    emb_layer = (model.bert.embeddings.word_embeddings
                 if hasattr(model, "bert")
                 else model.roberta.embeddings.word_embeddings)

    emb       = emb_layer(input_ids)
    emb       = emb.detach().requires_grad_(True)

    outputs  = model(inputs_embeds=emb, attention_mask=attention_mask)
    loss     = F.cross_entropy(outputs.logits, true_label)
    loss.backward()

    grad     = emb.grad.detach().squeeze(0)
    scores   = grad.norm(dim=-1).cpu()
    return scores


def _candidate_tokens_fgsm(
    model, input_ids, attn_mask, pos, label_t, device, top_k: int = 50
) -> list:
    """
    HotFlip-style FGSM candidates: rank vocabulary tokens by dot-product with
    the loss gradient at position `pos`.  Returns top-k token IDs (excluding
    the original token) that maximally increase cross-entropy in one step.

    Reference: Ebrahimi et al. "HotFlip: White-Box Adversarial Examples for
    Text Classification", ACL 2018.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    emb_layer = (model.bert.embeddings.word_embeddings
                 if hasattr(model, "bert")
                 else model.roberta.embeddings.word_embeddings)

    emb = emb_layer(input_ids).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=attn_mask)
    loss = F.cross_entropy(out.logits, label_t)
    loss.backward()

    grad_at_pos = emb.grad[0, pos].detach()
    all_embs    = emb_layer.weight.detach()
    scores  = torch.mv(all_embs, grad_at_pos)
    top_ids = scores.topk(top_k + 1).indices.tolist()

    orig_id = input_ids[0, pos].item()
    return [tid for tid in top_ids if tid != orig_id][:top_k]


def whitebox_attack(
    model,
    tokenizer,
    claim:      str,
    true_label: int,
    attack_type: str,
    epsilon:    float,
    n_steps:    int,
    device:     str,
) -> str:
    """
    Gradient-guided adversarial substitution attack.

    epsilon is interpreted as the fraction of non-special tokens to replace.
    Returns adversarially perturbed text.
    """
    import torch

    model.eval()

    enc         = tokenizer(claim, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
    input_ids   = enc["input_ids"]
    attn_mask   = enc["attention_mask"]
    seq_len     = input_ids.shape[1]
    label_t     = torch.tensor([true_label], device=device)

    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                   tokenizer.pad_token_id}

    perturb_positions = [
        i for i in range(seq_len)
        if input_ids[0, i].item() not in (special_ids or set())
    ]
    n_to_replace = max(1, int(len(perturb_positions) * epsilon))

    ids_working  = input_ids.clone()
    iterations   = n_steps if attack_type == "pgd" else 1

    for _ in range(iterations):
        scores   = _token_importance(
            model, tokenizer, ids_working, attn_mask, label_t, device
        )
        eligible_scores = [(scores[pos].item(), pos) for pos in perturb_positions]
        eligible_scores.sort(reverse=True)
        top_positions = [pos for _, pos in eligible_scores[: n_to_replace]]

        for pos in top_positions:
            orig_id   = ids_working[0, pos].item()
            best_id   = orig_id
            best_loss = -float("inf")

            candidates = _candidate_tokens_fgsm(
                model, ids_working, attn_mask, pos, label_t, device, top_k=50
            )
            for cand_id in candidates:
                if cand_id == orig_id:
                    continue
                try_ids        = ids_working.clone()
                try_ids[0, pos] = cand_id

                with torch.no_grad():
                    import torch.nn.functional as F
                    out  = model(input_ids=try_ids, attention_mask=attn_mask)
                    loss = F.cross_entropy(out.logits, label_t).item()

                if loss > best_loss:
                    best_loss = loss
                    best_id   = cand_id

            ids_working[0, pos] = best_id

    return tokenizer.decode(ids_working[0], skip_special_tokens=True)



def append_csv(path: str, row: dict, fieldnames: list) -> None:
    p            = Path(path)
    write_header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


RESULTS_FIELDS = [
    "dataset", "model", "attack_type", "epsilon", "seed",
    "vs", "ir", "csr", "csr_ir", "n_target_correct", "n_targets", "n_perturbed",
    "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
]
TIMING_FIELDS = [
    "dataset", "model", "attack_type", "epsilon", "seed",
    "time_condition_s", "time_search_s", "time_eval_s", "n_candidates",
]



def main():
    args         = parse_args()
    t0_condition = time.time()
    rng          = random.Random(args.seed)

    experiments_dir = Path(__file__).parent
    project_root    = experiments_dir.parent
    sys.path.insert(0, str(experiments_dir))

    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[eval_whitebox] WARNING: running on CPU — white-box attacks are very slow. "
            "Consider using --max_samples 50 for smoke-testing.",
            file=sys.stderr,
        )

    from data_loaders import load_dataset_split
    _, _, test_data = load_dataset_split(args.dataset, seed=args.seed)

    if args.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: args.max_samples]

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(
            f"[eval_whitebox] Checkpoint not found at {ckpt_dir}. "
            "Train the model first using eval_structured.py with bert/roberta. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()
    print(f"[eval] Loaded {args.model_name} from {ckpt_dir}  device={device}")

    n_test   = len(test_data)
    n_target = max(1, int(args.budget * n_test))
    idx      = list(range(n_test))
    rng.shuffle(idx)
    target_idx    = set(idx[: n_target])
    target_claims = [test_data[i] for i in sorted(target_idx)]

    print(
        f"[eval] {args.dataset}  model={args.model_name}  "
        f"attack={args.attack_type}  eps={args.epsilon}  "
        f"n_test={n_test}  n_targets={n_target}"
    )

    t0_search = time.time()
    adv_lookup = {}
    for i, rec in enumerate(target_claims):
        print(
            f"  [{i+1}/{n_target}] {args.attack_type} eps={args.epsilon} "
            f"| {rec['claim'][:60]}...",
            end=" ",
        )
        t_s = time.time()
        adv = whitebox_attack(
            model, tokenizer,
            claim       = rec["claim"],
            true_label  = rec["label"],
            attack_type = args.attack_type,
            epsilon     = args.epsilon,
            n_steps     = args.n_steps,
            device      = device,
        )
        elapsed = round(time.time() - t_s, 1)
        print(f"→ {elapsed}s")
        adv_lookup[rec["claim"]] = adv

    t_search = round(time.time() - t0_search, 2)
    print(f"  ↳ Attack generation: {t_search:.1f}s for {len(adv_lookup)} claims")

    import torch.nn.functional as F

    def _predict_batch(records: list) -> list:
        preds    = []
        batch_sz = 8
        for i in range(0, len(records), batch_sz):
            batch = [r["claim"] for r in records[i: i + batch_sz]]
            enc   = tokenizer(
                batch, truncation=True, padding=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
        return preds

    t0_eval = time.time()
    adv_test = []
    for i, rec in enumerate(test_data):
        if i in target_idx and rec["claim"] in adv_lookup:
            adv_test.append({"claim": adv_lookup[rec["claim"]], "label": rec["label"]})
        else:
            adv_test.append({"claim": rec["claim"], "label": rec["label"]})

    preds_clean = _predict_batch(test_data)
    preds_adv   = _predict_batch(adv_test)
    t_eval      = round(time.time() - t0_eval, 2)

    n_nonT = n_test - n_target
    lab    = [r["label"] for r in test_data]

    vs_hits   = sum(int(preds_clean[i] == lab[i] and preds_adv[i] != lab[i]) for i in target_idx)
    ir_flips  = sum(int(preds_adv[i] != preds_clean[i]) for i in target_idx)
    n_target_correct = sum(int(preds_clean[i] == lab[i]) for i in target_idx)
    csr_hits  = sum(int(preds_clean[i] == lab[i] and preds_adv[i] != lab[i])
                    for i in range(n_test) if i not in target_idx)
    csr_flips = sum(int(preds_adv[i] != preds_clean[i])
                    for i in range(n_test) if i not in target_idx)
    n_nonT_correct = sum(int(preds_clean[i] == lab[i])
                         for i in range(n_test) if i not in target_idx)

    vs          = round(vs_hits  / max(n_target_correct, 1), 4)
    ir          = round(ir_flips / max(n_target, 1), 4)
    csr         = round(csr_hits / max(n_nonT_correct, 1), 4)
    csr_ir      = round(csr_flips / max(n_nonT, 1), 4)
    t_condition = round(time.time() - t0_condition, 2)

    print(f"  ↳ Eval: {t_eval:.1f}s  VS={vs:.4f}  IR={ir:.4f}  CSR={csr:.4f}")
    print(
        f"\n  {args.dataset} | {args.model_name} | {args.attack_type} eps={args.epsilon}"
        f" | search={t_search:.1f}s eval={t_eval:.1f}s total={t_condition:.1f}s"
    )

    append_csv(args.output_csv, {
        "dataset":          args.dataset,
        "model":            args.model_name,
        "attack_type":      args.attack_type,
        "epsilon":          args.epsilon,
        "seed":             args.seed,
        "vs":               vs,
        "ir":               ir,
        "csr_ir":           csr_ir,
        "n_target_correct": n_target_correct,
        "csr":              csr,
        "n_targets":        n_target,
        "n_perturbed":      len(adv_lookup),
        "time_search_s":    t_search,
        "time_eval_s":      t_eval,
        "n_candidates":     50 * len(adv_lookup),
        "time_condition_s": t_condition,
    }, RESULTS_FIELDS)

    append_csv(args.timing_csv, {
        "dataset":          args.dataset,
        "model":            args.model_name,
        "attack_type":      args.attack_type,
        "epsilon":          args.epsilon,
        "seed":             args.seed,
        "time_condition_s": t_condition,
        "time_search_s":    t_search,
        "time_eval_s":      t_eval,
        "n_candidates":     50 * len(adv_lookup),
    }, TIMING_FIELDS)

    print(
        f"[eval_whitebox] {args.dataset} | {args.model_name} | "
        f"{args.attack_type} eps={args.epsilon} | VS={vs:.4f} | {t_condition:.1f}s  ✓"
    )


if __name__ == "__main__":
    main()
