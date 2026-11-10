#!/usr/bin/env python3
"""
eval_structured.py — C8: SLM/LLM vulnerability on structured datasets.

Evaluates LSTM, BERT, RoBERTa, GPT-4, and Llama-3 victims on structured
fact-checking datasets (weather, population, flights) under three attack
types: sybil, content_perturbation, llm_generated / prompt_injection.

Attack mapping for LLM/SLM victims:
  - DL models (lstm_title):         sybil, content_perturbation, llm_generated
  - SLM models (bert, roberta):     same
  - LLM models (gpt4_zeroshot,
                llama3_zeroshot):   sybil → content_perturbation (proxy),
                                    llm_generated → prompt_injection

Usage (matches run_exp_C8.sh):
    python eval_structured.py \
        --dataset weather --model bert \
        --attack_type sybil \
        --budget 0.10 --seed 42 \
        --output_csv results.csv --timing_csv timing.csv

API keys (LLM models): GROQ_API_KEY or OPENAI_API_KEY.

Last updated: 2026-04-16
"""

import argparse
import csv
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True,
                   choices=["weather", "population", "flights", "tabfact",
                            "covid19", "pubhealth", "kaggle"])
    p.add_argument("--model",        required=True,
                   choices=["lstm_title", "bert", "roberta",
                             "gpt4_zeroshot", "llama3_zeroshot"])
    p.add_argument("--attack_type",  required=True,
                   choices=["sybil", "content_perturbation",
                             "llm_generated", "prompt_injection"])
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
    p.add_argument("--max_samples",  type=int, default=None)
    p.add_argument("--checkpoint_dir", default=None,
                   help="Override the SLM checkpoint directory (bert/roberta only).")
    args = p.parse_args()
    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



_NUMERALS_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _perturb_number(match):
    """Scale a matched numeric string by a random ±20–60% factor."""
    original = float(match.group().replace(",", ""))
    factor   = 1.0 + random.choice([-0.5, -0.3, 0.3, 0.6])
    return str(round(original * factor, 1))


def content_perturbation(claims: list, rng: random.Random) -> list:
    """
    Structured content perturbation: replace numeric values in the claim
    text with implausible alternatives (±20–60% distortion).
    """
    results = []
    for item in claims:
        text       = item["claim"]
        adversarial = _NUMERALS_RE.sub(lambda m: _perturb_number(m), text)
        if adversarial == text:
            adversarial = "Contrary to official records, " + text
        results.append({
            "original":    text,
            "adversarial": adversarial,
            "label":       item["label"],
            "attack_type": "content_perturbation",
        })
    return results


def sybil_attack(claims: list, rng: random.Random, n_sybil: int = 3) -> list:
    """
    Sybil poisoning proxy for the claim-level evaluation:
    replaces target claim text with n_sybil repetitions of a
    false assertion, simulating Sybil-boosted misinformation.
    The adversarial claim is a content-perturbed version labelled
    as confident misinformation.
    """
    results = []
    for item in claims:
        text       = item["claim"]
        perturbed  = _NUMERALS_RE.sub(lambda m: _perturb_number(m), text)
        adversarial = " ".join([perturbed] * min(n_sybil, 2))
        results.append({
            "original":    text,
            "adversarial": adversarial,
            "label":       item["label"],
            "attack_type": "sybil",
        })
    return results


def prompt_injection(claims: list, rng: random.Random) -> list:
    """Delegate to the canonical generator in attack_generators.

    This function used to define its own three-template suffix set, with different wording from the
    five templates the paper reports, and to substitute FALSE where the canonical set substitutes
    FAKE. No reported number is affected -- prompt injection was never run against a structured
    victim, and no c8_* result file contains it -- but an artifact user running
    `--attack_type prompt_injection` would have got an attack the paper never describes, from a
    module that already imported the real one.
    """
    from attack_generators import generate_prompt_injection
    return generate_prompt_injection(claims, rng)



def predict_dl(model, vocab, max_len: int, records: list, device: str) -> list:
    from dl_models import predict_batch
    return predict_batch(model, records, vocab, max_len, device)


def _encode_args(records: list) -> tuple:
    """Tokenizer positional args: (claims,) or (claims, evidence) for evidence-bearing corpora.

    TabFact is table-based verification — the claim alone carries no evidence, and a victim given
    only the claim cannot beat the majority-class baseline. Where the corpus supplies evidence we
    encode it as the second segment of a sentence pair, so the model sees claim [SEP] table.

    Evidence is passed separately rather than concatenated into the claim string on purpose: the
    attacks in this benchmark perturb `claim`, and folding the table into that field would let an
    attack rewrite the very evidence it is supposed to be checked against.

    All-or-nothing by design. A batch where only some items carry evidence would train the model to
    read the second segment's presence as a label hint, so a partially-populated corpus falls back
    to claim-only and says so.
    """
    claims = [r["claim"] for r in records]
    ev = [r.get("evidence", "") for r in records]
    if all(ev):
        return claims, ev
    if any(ev):
        warnings.warn(
            f"[eval] evidence present on only {sum(1 for e in ev if e)}/{len(ev)} items; "
            f"encoding claim-only to avoid making evidence-presence a label cue",
            UserWarning, stacklevel=2,
        )
    return (claims,)


def predict_slm(model_name: str, tokenizer, slm_model, records: list,
                device: str, max_length: int = 512) -> list:
    """BERT/RoBERTa fine-tuned classification."""
    import torch
    preds   = []
    batch_sz = 16
    for i in range(0, len(records), batch_sz):
        batch  = records[i: i + batch_sz]
        enc    = tokenizer(*_encode_args(batch), truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = slm_model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def predict_llm_api(client, model_id: str, records: list,
                    shot_mode: str = "zero_shot") -> list:
    """GPT-4 / Llama-3 via OpenAI-compatible API."""
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_llm_victim import build_prompt_messages, parse_llm_label

    preds = []
    for rec in records:
        msgs  = build_prompt_messages(rec["claim"], shot_mode)
        label = None
        for attempt in range(3):
            try:
                resp  = client.chat.completions.create(
                    model=model_id, messages=msgs, max_tokens=10, temperature=0.0,
                )
                label = parse_llm_label(resp.choices[0].message.content)
                if label is not None:
                    break
            except Exception:
                import time as _t; _t.sleep(2 ** attempt)
        preds.append(label if label is not None else -1)
    return preds



def load_or_train_slm(model_name: str, dataset: str, project_root: Path,
                      train_data: list, val_data: list,
                      device: str, seed: int,
                      max_train: int = 5000, num_epochs: int = 5,
                      learning_rate: float = 2e-5,
                      ckpt_dir: Path | None = None):
    """Load fine-tuned BERT/RoBERTa or fine-tune from scratch.

    `learning_rate` and `ckpt_dir` exist for the training-configuration sensitivity sweep
    over training configuration. Their
    defaults reproduce the configuration the published checkpoints were trained under —
    lr 2e-5, batch 32, 5 epochs, weight decay 0.01 — so leaving them unset changes nothing.
    """
    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer, DataCollatorWithPadding)
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        raise ImportError("pip install transformers torch scikit-learn")

    if ckpt_dir is None:
        ckpt_dir = project_root / "checkpoints" / f"{model_name}_{dataset}"

    ckpt_valid = ckpt_dir.exists() and (ckpt_dir / "config.json").exists()
    if ckpt_valid:
        print(f"[eval] Loading SLM checkpoint: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model     = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir)).to(device)
        model.eval()
        return tokenizer, model

    if ckpt_dir.exists() and not ckpt_valid:
        print(f"[eval] Checkpoint dir exists but is empty/corrupt — retraining: {ckpt_dir}")

    if len(train_data) > max_train:
        import random as _random
        rng = _random.Random(seed)
        train_data = rng.sample(train_data, max_train)
        print(f"[eval] Training data capped at {max_train}")

    print(f"[eval] Fine-tuning {model_name} on {dataset} ({len(train_data)} train rows)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    class _DS(TorchDataset):
        def __init__(self, data):
            enc = tokenizer(
                *_encode_args(data),
                truncation=True, padding=False, max_length=512,
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.token_type_ids = enc.get("token_type_ids")
            self.labels         = [int(r["label"]) for r in data]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {"input_ids": self.input_ids[i],
                    "attention_mask": self.attention_mask[i],
                    "labels": self.labels[i]}
            if self.token_type_ids is not None:
                item["token_type_ids"] = self.token_type_ids[i]
            return item

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    total_steps = max(1, (len(train_data) // 32) * num_epochs)
    warmup = min(500, max(10, total_steps // 10))
    tr_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_epochs, per_device_train_batch_size=32,
        learning_rate=learning_rate, weight_decay=0.01, warmup_steps=warmup,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        seed=seed, no_cuda=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=_DS(train_data),
        eval_dataset=_DS(val_data),
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    model.eval()
    return tokenizer, model



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
    "dataset", "model", "attack_type", "seed",
    "metric", "metric_value", "label_semantics", "baseline",
    "vs", "dfr", "ir", "csr", "csr_ir", "n_target_correct", "accuracy_clean", "accuracy_poisoned",
    "n_sources", "n_malicious",
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
    random.seed(args.seed)

    experiments_dir = Path(__file__).parent
    project_root    = experiments_dir.parent
    sys.path.insert(0, str(experiments_dir))

    from data_loaders import load_dataset_split
    train_data, val_data, test_data = load_dataset_split(args.dataset, seed=args.seed)

    if args.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: args.max_samples]

    n_test   = len(test_data)
    n_target = max(1, int(args.budget * n_test))
    all_idx  = list(range(n_test))
    rng.shuffle(all_idx)
    target_idx    = set(all_idx[: n_target])
    target_claims = [test_data[i] for i in sorted(target_idx)]

    print(
        f"[eval] {args.dataset}  model={args.model}  "
        f"attack={args.attack_type}  n_test={n_test}  n_targets={n_target}"
    )

    t0_search = time.time()
    if args.attack_type == "sybil":
        adversarial_pairs = sybil_attack(target_claims, rng)
    elif args.attack_type == "content_perturbation":
        adversarial_pairs = content_perturbation(target_claims, rng)
    elif args.attack_type == "prompt_injection":
        adversarial_pairs = prompt_injection(target_claims, rng)
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
    print(f"  ↳ Attack generation: {t_search:.1f}s")

    adv_lookup = {p["original"]: p["adversarial"] for p in adversarial_pairs}
    adv_test   = []
    for i, rec in enumerate(test_data):
        item = dict(rec)
        if i in target_idx and rec["claim"] in adv_lookup:
            item["claim"] = adv_lookup[rec["claim"]]
        adv_test.append(item)

    t0_eval = time.time()
    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "lstm_title":
        from dl_models import MODEL_CONFIGS, load_model, train_model, predict_batch

        ckpt_dir = str(project_root / "checkpoints" / f"lstm_title_{args.dataset}")
        if (Path(ckpt_dir) / "model.pt").exists():
            model, vocab = load_model("lstm_title", ckpt_dir, device)
        else:
            model, vocab = train_model(
                "lstm_title", train_data, val_data, ckpt_dir,
                device=device, seed=args.seed,
            )
        max_len    = MODEL_CONFIGS["lstm_title"]["max_len"]
        preds_c    = predict_batch(model, test_data, vocab, max_len, device)
        preds_a    = predict_batch(model, adv_test,  vocab, max_len, device)

    elif args.model in ("bert", "roberta"):
        hf_name   = "bert-base-uncased" if args.model == "bert" else "roberta-base"
        tokenizer, slm_model = load_or_train_slm(
            hf_name, args.dataset, project_root,
            train_data, val_data, device, args.seed,
            ckpt_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        )
        preds_c = predict_slm(hf_name, tokenizer, slm_model, test_data, device)
        preds_a = predict_slm(hf_name, tokenizer, slm_model, adv_test,  device)

    else:
        from openai import OpenAI

        if args.model == "gpt4_zeroshot":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Set OPENAI_API_KEY for gpt4_zeroshot")
            client   = OpenAI(api_key=api_key, timeout=180.0, max_retries=3)
            model_id = "gpt-4o-mini"
        else:
            api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Set GROQ_API_KEY or OPENAI_API_KEY for llama3_zeroshot")
            base_url = ("https://api.groq.com/openai/v1"
                        if os.environ.get("GROQ_API_KEY") else None)
            kwargs   = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client   = OpenAI(**kwargs, timeout=180.0, max_retries=3)
            model_id = "llama-3.3-70b-versatile"

        preds_c = predict_llm_api(client, model_id, test_data)
        preds_a = predict_llm_api(client, model_id, adv_test)

    t_eval = round(time.time() - t0_eval, 2)

    n_nonT         = n_test - n_target
    lab            = [r["label"] for r in test_data]

    from attack_semantics import (gold_label_after, majority_baseline,
                                  metric_for)
    gold_after = {i: gold_label_after(args.attack_type, lab[i]) for i in target_idx}
    vs_hits        = sum(int(preds_c[i] == lab[i] and preds_a[i] != gold_after[i])
                         for i in target_idx)
    ir_flips       = sum(int(preds_a[i] != preds_c[i]) for i in target_idx)
    n_target_correct = sum(int(preds_c[i] == lab[i]) for i in target_idx)
    target_gold_scored = [lab[i] for i in target_idx if preds_c[i] == lab[i]]

    csr_hits       = sum(int(preds_c[i] == lab[i] and preds_a[i] != lab[i])
                         for i in range(n_test) if i not in target_idx)
    csr_flips      = sum(int(preds_a[i] != preds_c[i])
                         for i in range(n_test) if i not in target_idx)
    n_nonT_correct = sum(int(preds_c[i] == lab[i])
                         for i in range(n_test) if i not in target_idx)

    correct_clean  = sum(int(preds_c[i] == lab[i]) for i in range(n_test))
    correct_poison = sum(int(preds_a[i] == gold_after.get(i, lab[i])) for i in range(n_test))

    metric           = metric_for(args.attack_type)
    rate             = round(vs_hits / max(n_target_correct, 1), 4)
    baseline         = majority_baseline(args.attack_type, target_gold_scored)
    vs               = rate if metric == "vs" else None
    dfr              = rate if metric == "dfr" else None
    ir               = round(ir_flips / max(n_target, 1), 4)
    csr              = round(csr_hits / max(n_nonT_correct, 1), 4)
    csr_ir           = round(csr_flips / max(n_nonT, 1), 4)
    acc_clean        = round(correct_clean  / n_test, 4)
    acc_poisoned     = round(correct_poison / n_test, 4)
    t_condition      = round(time.time() - t0_condition, 2)
    n_malicious_src  = int(args.budget * n_test)

    print(
        f"  ↳ {metric.upper()}={rate:.4f}"
        f"{'' if baseline is None else f' (trivial {baseline:.4f})'}"
        f"  IR={ir:.4f}  CSR={csr:.4f}  "
        f"acc_clean={acc_clean:.4f}  acc_poisoned={acc_poisoned:.4f}"
    )
    print(
        f"\n  {args.dataset} | {args.model} | {args.attack_type}"
        f" | search={t_search:.1f}s eval={t_eval:.1f}s total={t_condition:.1f}s"
    )

    append_csv(args.output_csv, {
        "dataset":          args.dataset,
        "model":            args.model,
        "attack_type":      args.attack_type,
        "seed":             args.seed,
        "metric":           metric,
        "metric_value":     rate,
        "label_semantics":  "preserving" if metric == "vs" else "changing",
        "baseline":         baseline,
        "vs":               vs,
        "dfr":              dfr,
        "ir":               ir,
        "csr":              csr,
        "csr_ir":           csr_ir,
        "n_target_correct": n_target_correct,
        "accuracy_clean":   acc_clean,
        "accuracy_poisoned": acc_poisoned,
        "n_sources":        n_test,
        "n_malicious":      n_malicious_src,
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
        f"[eval_structured] {args.dataset} | {args.model} | {args.attack_type}"
        f" | {metric.upper()}={rate:.4f} | {t_condition:.1f}s  ✓"
    )


if __name__ == "__main__":
    main()
