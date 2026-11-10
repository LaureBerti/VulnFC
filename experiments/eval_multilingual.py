#!/usr/bin/env python3
"""
eval_multilingual.py — C5: Cross-lingual attack transfer (English → French, Arabic).

Tests whether adversarial attacks generated in English transfer to French and Arabic
on multilingual fact-checking datasets using multilingual transformer models
(mBERT, XLM-RoBERTa). Appends one row to --output_csv and --timing_csv per
invocation.

Checkpoint policy: loads from checkpoints/{model_safe}_{language}_{dataset}/
if config.json is present; fine-tunes from scratch and saves if absent.

Usage (matches run_exp_C5.sh):
    python eval_multilingual.py \
        --language en \
        --model bert-base-multilingual-cased \
        --attack_type llm_generated \
        --attack_language en \
        --pos_tagger multilingual \
        --budget 0.10 \
        --seed 42 \
        --dataset_name xfact \
        --output_csv results.csv \
        --timing_csv timing.csv

Primary dataset: XFact (Gupta et al. 2021) — multilingual fact-checking.
Fallback: load_dataset_split() from data_loaders.py (synthetic).

API keys (for llm_generated only): set GROQ_API_KEY or OPENAI_API_KEY.
If neither is set, falls back to negation attack with a warning.

Last updated: 2026-04-17
"""

import argparse
import csv
import os
import re
import random
import sys
import time
import warnings
from pathlib import Path



def parse_args():
    p = argparse.ArgumentParser(
        description="C5: Cross-lingual attack transfer on multilingual fact-checking."
    )
    p.add_argument("--language",        required=True,
                   choices=["en", "fr", "ar"],
                   help="Target language for the victim model and test data")
    p.add_argument("--model",           required=True,
                   choices=["bert-base-multilingual-cased",
                             "xlm-roberta-base",
                             "bert-base-uncased"],
                   help="Multilingual (or EN-only baseline) transformer model")
    p.add_argument("--attack_type",     required=True,
                   choices=["llm_generated", "negation"],
                   help="Attack type to evaluate")
    p.add_argument("--attack_language", default=None,
                   choices=["en", "fr", "ar", None],
                   help="Language in which attacks are generated (default: same as --language)")
    p.add_argument("--pos_tagger",      default=None,
                   help="POS tagger variant (e.g. 'multilingual'); informational only")
    p.add_argument("--budget",          type=float, default=0.10,
                   help="Fraction of test set to target (default 0.10)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--dataset_name",    default="xfact",
                   help="Dataset name (default: xfact)")
    p.add_argument("--output_csv",      required=True)
    p.add_argument("--timing_csv",      required=True)
    p.add_argument("--max_samples",     type=int, default=None,
                   help="Cap test set size for smoke-testing")
    p.add_argument("--api_base",        default=None,
                   help="Override LLM API base URL (e.g. http://localhost:11434/v1 for Ollama)")
    p.add_argument("--attack_model",    default=None,
                   help="Override LLM model name for attack generation (e.g. llama3.2:latest)")
    args = p.parse_args()

    if args.attack_language is None:
        args.attack_language = args.language

    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



_AUX_RE = re.compile(
    r"\b(is|are|was|were|will|can|should|would|could|must|has|have|had|do|does|did)\b",
    re.IGNORECASE,
)

_NEGATION_PREFIX = {
    "en": ("It is not true that ", True),
    "fr": ("Il n'est pas vrai que ", False),
    "ar": ("ليس صحيحاً أن ", False),
}


def _negate(text: str, language: str = "en") -> str:
    """
    Insert 'not' after the first auxiliary verb (English pattern).
    For non-English, if no auxiliary found, prepend a language-aware negation phrase.
    """
    m = _AUX_RE.search(text)
    if m:
        return text[: m.end()] + " not" + text[m.end():]
    prefix, lower_first = _NEGATION_PREFIX.get(language, _NEGATION_PREFIX["en"])
    if lower_first and text:
        return prefix + text[0].lower() + text[1:]
    return prefix + text


def generate_negation(claims: list, language: str = "en") -> list:
    return [
        {
            "original":    item["claim"],
            "adversarial": _negate(item["claim"], language),
            "label":       item["label"],
            "attack_type": "negation",
        }
        for item in claims
    ]



_XFACT_LABEL_MAP = {
    "true":         1,
    "false":        0,
    "mostly true":  1,
    "mostly false": 0,
}


def _binarise_xfact(raw) -> "int | None":
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return _XFACT_LABEL_MAP.get(s)


def _read_xfact_tsv(path: Path, language: str = None) -> list:
    """
    Read one XFact TSV file (tab-separated, header row).
    Columns: language, site, evidence_1..5, link_1..5, claimDate,
             reviewDate, claimant, claim, label
    If language is given, filter to that language code.
    Returns list of {"claim": str, "label": int}.
    """
    import csv as _csv
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for row in reader:
            if language and row.get("language", "").strip().lower() != language.lower():
                continue
            text = str(row.get("claim", "")).strip()
            if not text:
                continue
            label = _binarise_xfact(row.get("label", ""))
            if label is None:
                continue
            records.append({"claim": text, "label": label})
    return records


def _xfact_resplit(records: list, seed: int) -> "tuple[list, list, list]":
    """Stratified 70/10/20 split for when XFact natural splits aren't available."""
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    def _do(items):
        s = items[:]
        rng.shuffle(s)
        n_tr = int(0.70 * len(s))
        n_va = int(0.10 * len(s))
        return s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]

    tr_p, va_p, te_p = _do(pos)
    tr_n, va_n, te_n = _do(neg)

    def merge(a, b):
        m = a + b
        rng.shuffle(m)
        return m

    return merge(tr_p, tr_n), merge(va_p, va_n), merge(te_p, te_n)


def load_xfact(language: str, seed: int) -> "tuple[list, list, list]":
    """
    Load XFact from local TSV files (data/xfact/).
    Data source: https://github.com/utahnlp/x-fact

    Language coverage:
      - ar (and others with train/dev/test): uses natural XFact splits
      - en: x-fact-including-en/train.all.tsv filtered to en; re-split 70/10/20
      - fr (zero-shot): zeroshot.tsv for test; all other-language train data for
        training (cross-lingual transfer setup)

    Returns (train, val, test) as lists of {"claim": str, "label": int}.
    """
    experiments_dir = Path(__file__).parent
    data_dir = experiments_dir.parent / "data" / "xfact"

    xfact_dir    = data_dir / "x-fact"
    xfact_en_dir = data_dir / "x-fact-including-en"

    if not xfact_dir.exists():
        raise FileNotFoundError(
            f"XFact data not found at {xfact_dir}. "
            "Download from https://github.com/utahnlp/x-fact and place under data/xfact/"
        )

    lang = language.lower()

    if lang == "en":
        train_file = xfact_en_dir / "train.all.tsv"
        if not train_file.exists():
            raise FileNotFoundError(f"Expected {train_file}")
        all_en = _read_xfact_tsv(train_file, language="en")
        return _xfact_resplit(all_en, seed)

    if lang == "fr":
        train_all = _read_xfact_tsv(xfact_dir / "train.all.tsv", language=None)
        dev_data  = _read_xfact_tsv(xfact_dir / "dev.all.tsv",   language=None)
        zeroshot_file = xfact_dir / "zeroshot.tsv"
        fr_all = _read_xfact_tsv(zeroshot_file, language="fr") if zeroshot_file.exists() else []
        if not fr_all:
            raise RuntimeError("No French zero-shot data found in data/xfact/x-fact/zeroshot.tsv")
        rng = random.Random(seed)
        rng.shuffle(fr_all)
        n_tr = int(0.50 * len(fr_all))
        n_va = int(0.10 * len(fr_all))
        fr_train = fr_all[:n_tr]
        fr_val   = fr_all[n_tr:n_tr + n_va]
        fr_test  = fr_all[n_tr + n_va:]
        train_data = train_all + fr_train
        return train_data, dev_data + fr_val, fr_test

    train_file = xfact_dir / "train.all.tsv"
    dev_file   = xfact_dir / "dev.all.tsv"
    test_file  = xfact_dir / "test.all.tsv"

    train_data = _read_xfact_tsv(train_file, language=lang) if train_file.exists() else []
    dev_data   = _read_xfact_tsv(dev_file,   language=lang) if dev_file.exists()   else []
    test_data  = _read_xfact_tsv(test_file,  language=lang) if test_file.exists()  else []

    if not train_data and not test_data:
        zeroshot_file = xfact_dir / "zeroshot.tsv"
        all_records = _read_xfact_tsv(zeroshot_file, language=lang) if zeroshot_file.exists() else []
        if not all_records:
            raise RuntimeError(
                f"No XFact data found for language='{lang}'. "
                "Check data/xfact/ contains x-fact/train.all.tsv and x-fact/zeroshot.tsv."
            )
        return _xfact_resplit(all_records, seed)

    if train_data and not dev_data:
        return _xfact_resplit(train_data + test_data, seed)

    return train_data, dev_data, test_data



def _safe_model_name(model_name: str) -> str:
    """Convert 'bert-base-multilingual-cased' → 'bert-base-multilingual-cased'."""
    return model_name.replace("/", "_").replace("\\", "_")


def _ckpt_dir(project_root: Path, model_name: str, language: str, dataset: str) -> Path:
    safe = _safe_model_name(model_name)
    return project_root / "checkpoints" / f"{safe}_{language}_{dataset}"


def _records_to_hf_dataset(records: list):
    """Convert [{"claim": str, "label": int}] to a HuggingFace Dataset."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict(
        {
            "text":  [r["claim"] for r in records],
            "label": [r["label"] for r in records],
        }
    )


def load_or_train_transformer(
    model_name:   str,
    train_data:   list,
    val_data:     list,
    ckpt_dir:     Path,
    device:       str,
    seed:         int,
):
    """
    Load a fine-tuned HuggingFace transformer from checkpoint if available,
    otherwise fine-tune from pretrained and save.

    Returns (model, tokenizer).
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
    except ImportError:
        raise ImportError(
            "pip install transformers>=4.44.0 accelerate  # required for transformer models"
        )

    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _load_dir = None
    if (ckpt_dir / "config.json").exists():
        _load_dir = ckpt_dir
    else:
        _subdirs = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        ) if ckpt_dir.exists() else []
        if _subdirs:
            _load_dir = _subdirs[-1]

    if _load_dir is not None:
        print(f"[eval] Loading checkpoint: {_load_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(_load_dir))
        model.to("cpu")
        model.eval()
        return model, tokenizer

    print(f"[eval] No checkpoint at {ckpt_dir} — fine-tuning {model_name} ...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    model.to(device)

    max_length = 128

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    hf_train = _records_to_hf_dataset(train_data).map(_tokenize, batched=True)
    hf_val   = _records_to_hf_dataset(val_data).map(_tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in hf_train.column_names:
        cols.append("token_type_ids")
    hf_train.set_format("torch", columns=cols)
    hf_val.set_format("torch", columns=cols)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val if hf_val else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"[eval] Checkpoint saved: {ckpt_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    model.to("cpu")
    model.eval()
    return model, tokenizer


def predict_transformer(model, tokenizer, records: list, device: str) -> list:
    """
    Run inference on a list of {"claim": str, ...} records.
    Returns list of int predictions (0/1).
    """
    import torch

    texts = [r["claim"] for r in records]
    preds = []

    batch_size = 32
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            batch_preds = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_preds)
    return preds



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
    "language", "model", "attack_type", "budget",
    "accuracy_pre", "accuracy_post",
    "vs", "csr",
    "vs_delta_from_en",
    "test_pos_frac", "reliability",
    "time_attack_gen_s", "time_eval_s", "n_candidates", "time_condition_s",
]

TIMING_FIELDS = [
    "language", "model", "attack_type",
    "time_condition_s", "time_attack_gen_s", "time_eval_s", "n_candidates",
]



def main():
    args         = parse_args()
    t0_condition = time.time()
    rng          = random.Random(args.seed)

    experiments_dir = Path(__file__).parent
    project_root    = experiments_dir.parent
    sys.path.insert(0, str(experiments_dir))

    try:
        import transformers
    except ImportError:
        raise ImportError(
            "transformers is required: pip install transformers>=4.44.0 accelerate"
        )

    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"
    print(f"[eval] Device: {device}")

    MIN_BINARY_RECORDS = 50
    train_data = val_data = test_data = None

    if args.dataset_name == "xfact":
        try:
            train_data, val_data, test_data = load_xfact(args.language, seed=args.seed)
            n_all = len(train_data) + len(val_data) + len(test_data)
            if n_all < MIN_BINARY_RECORDS:
                warnings.warn(
                    f"[eval] XFact returned only {n_all} binary records for "
                    f"language='{args.language}'. Using synthetic fallback.",
                    UserWarning,
                )
                train_data = None
            else:
                print(
                    f"[eval] XFact: train={len(train_data)}, val={len(val_data)}, "
                    f"test={len(test_data)} (lang={args.language})"
                )
        except Exception as exc:
            warnings.warn(
                f"[eval] XFact load failed ({exc}). Falling back to climatefever.",
                UserWarning,
            )
            train_data = None

    if train_data is None:
        from data_loaders import load_dataset_split
        _fallback_ds = "climatefever"
        warnings.warn(
            f"[eval] Using '{_fallback_ds}' as proxy dataset. "
            f"Language filtering is not applied for proxy data. "
            f"Place XFact data at data/xfact/ for real multilingual results.",
            UserWarning,
        )
        train_data, val_data, test_data = load_dataset_split(
            _fallback_ds, seed=args.seed
        )

    if args.max_samples:
        rng.shuffle(test_data)
        test_data = test_data[: args.max_samples]
        print(f"[eval] Smoke-test: capped at {len(test_data)} samples")

    if len(test_data) < MIN_BINARY_RECORDS:
        warnings.warn(
            f"[eval] Only {len(test_data)} binary test records for "
            f"language='{args.language}'. Results may be unreliable.",
            UserWarning,
        )

    n_test = len(test_data)
    _pos_frac = sum(r["label"] for r in test_data) / max(n_test, 1)
    _imbalanced = _pos_frac > 0.85 or _pos_frac < 0.15
    print(
        f"[eval] language={args.language}  model={args.model}  "
        f"attack={args.attack_type}  n_test={n_test}  "
        f"pos_frac={_pos_frac:.2f}"
        + ("  [WARN: imbalanced — VS may be unreliable]" if _imbalanced else "")
    )

    ckpt = _ckpt_dir(project_root, args.model, args.language, args.dataset_name)

    if not val_data and train_data:
        split_idx = max(1, int(0.10 * len(train_data)))
        val_data   = train_data[-split_idx:]
        train_data = train_data[:-split_idx]

    model, tokenizer = load_or_train_transformer(
        model_name=args.model,
        train_data=train_data,
        val_data=val_data,
        ckpt_dir=ckpt,
        device=device,
        seed=args.seed,
    )

    n_target = max(1, int(args.budget * n_test))
    idx      = list(range(n_test))
    rng.shuffle(idx)
    target_idx    = set(idx[:n_target])
    target_claims = [test_data[i] for i in sorted(target_idx)]

    print(
        f"[eval] n_test={n_test}  n_targets={n_target}  "
        f"attack_language={args.attack_language}"
    )

    t0_attack = time.time()

    effective_attack = args.attack_type
    if args.attack_type == "llm_generated":
        api_key_available = bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        )
        if not api_key_available:
            warnings.warn(
                "[eval] OPENAI_API_KEY and GROQ_API_KEY are both unset. "
                "Falling back to negation attack for llm_generated.",
                UserWarning,
            )
            effective_attack = "negation"

    if effective_attack == "negation":
        adversarial_pairs = generate_negation(target_claims, language=args.attack_language)
    else:
        from attack_generators import generate_attacks
        adversarial_pairs = generate_attacks(
            claims=target_claims,
            attack_type="llm_generated",
            api_base=getattr(args, "api_base", None),
            attack_model=getattr(args, "attack_model", None),
            seed=args.seed,
        )

    t_attack_gen = round(time.time() - t0_attack, 2)
    n_candidates = len(adversarial_pairs)
    print(f"  ↳ Attack generation: {t_attack_gen:.1f}s  ({n_candidates} claims)")

    adv_lookup = {p["original"]: p["adversarial"] for p in adversarial_pairs}

    clean_records = test_data[:]
    adv_records   = []
    for i, rec in enumerate(clean_records):
        if i in target_idx and rec["claim"] in adv_lookup:
            adv_records.append({"claim": adv_lookup[rec["claim"]], "label": rec["label"]})
        else:
            adv_records.append({"claim": rec["claim"], "label": rec["label"]})

    t0_eval     = time.time()
    preds_clean = predict_transformer(model, tokenizer, clean_records, device)
    preds_adv   = predict_transformer(model, tokenizer, adv_records,   device)
    t_eval      = round(time.time() - t0_eval, 2)

    true_labels  = [r["label"] for r in clean_records]
    acc_pre  = round(
        sum(p == l for p, l in zip(preds_clean, true_labels)) / max(n_test, 1), 4
    )
    acc_post = round(
        sum(p == l for p, l in zip(preds_adv,   true_labels)) / max(n_test, 1), 4
    )

    n_nonT    = n_test - n_target
    vs_flips  = 0
    csr_flips = 0
    for i in range(n_test):
        flipped = int(preds_adv[i] != preds_clean[i])
        if i in target_idx:
            vs_flips  += flipped
        else:
            csr_flips += flipped

    vs          = round(vs_flips  / max(n_target, 1), 4)
    csr         = round(csr_flips / max(n_nonT,   1), 4)
    t_condition = round(time.time() - t0_condition, 2)

    print(f"  ↳ Eval: {t_eval:.1f}s  acc_pre={acc_pre:.4f}  acc_post={acc_post:.4f}")
    print(f"  ↳ VS={vs:.4f}  CSR={csr:.4f}")
    print(
        f"\n  {args.language} | {args.model} | {args.attack_type}"
        f" | attack_gen={t_attack_gen:.1f}s eval={t_eval:.1f}s total={t_condition:.1f}s"
    )

    append_csv(args.output_csv, {
        "language":          args.language,
        "model":             args.model,
        "attack_type":       args.attack_type,
        "budget":            args.budget,
        "accuracy_pre":      acc_pre,
        "accuracy_post":     acc_post,
        "vs":                vs,
        "csr":               csr,
        "vs_delta_from_en":  0.0,
        "test_pos_frac":     round(_pos_frac, 4),
        "reliability":       "low" if _imbalanced or n_target < 10 else "ok",
        "time_attack_gen_s": t_attack_gen,
        "time_eval_s":       t_eval,
        "n_candidates":      n_candidates,
        "time_condition_s":  t_condition,
    }, RESULTS_FIELDS)

    append_csv(args.timing_csv, {
        "language":          args.language,
        "model":             args.model,
        "attack_type":       args.attack_type,
        "time_condition_s":  t_condition,
        "time_attack_gen_s": t_attack_gen,
        "time_eval_s":       t_eval,
        "n_candidates":      n_candidates,
    }, TIMING_FIELDS)

    print(
        f"[eval_multilingual] {args.language} | {args.model} | {args.attack_type}"
        f" | VS={vs:.4f} | {t_condition:.1f}s  ✓"
    )


if __name__ == "__main__":
    main()
