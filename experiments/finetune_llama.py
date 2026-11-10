#!/usr/bin/env python3
"""
finetune_llama.py — LoRA fine-tuning of Llama-3-8B-Instruct for binary fact-checking.

Implements C2 fine-tuning step:
  - Loads dataset (pubhealth or politifact) training split
  - Applies LoRA (PEFT) via HuggingFace TRL SFTTrainer
  - Saves adapter weights to --output_dir
  - Logs training time for timing_per_condition.csv

Hardware: requires >= 40 GB VRAM (A100-80GB recommended).
          For smaller GPUs, reduce --batch_size and enable gradient_checkpointing.

Usage:
    python finetune_llama.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset pubhealth --split train --num_epochs 3 \
        --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
        --batch_size 8 --seed 42 \
        --output_dir outputs/c2_finetuned_llm/checkpoints/pubhealth_llama3ft

Last updated: 2026-04-16
"""

import argparse
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune Llama-3 for fact-checking")
    p.add_argument("--model_name",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset",       required=True, choices=["pubhealth", "politifact"])
    p.add_argument("--split",         default="train")
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--lora_r",        type=int,   default=16)
    p.add_argument("--lora_alpha",    type=int,   default=32)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--max_seq_len",   type=int,   default=512)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for gated models (or set HF_TOKEN env var)")
    args = p.parse_args()
    print(f"[config] {vars(args)}", file=sys.stderr)
    return args



SYSTEM_MSG = (
    "You are a fact-checking assistant. "
    "Given a claim, classify it as TRUE or FALSE. "
    "Output only TRUE or FALSE."
)


def record_to_chat(record: dict) -> str:
    """
    Convert {claim, label} to Llama-3 instruction format string for SFTTrainer.
    """
    label_str = "TRUE" if record["label"] == 1 else "FALSE"
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_MSG}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Claim: {record['claim']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{label_str}<|eot_id|>"
    )



def main():
    args = parse_args()
    t0   = time.time()

    _hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        try:
            import huggingface_hub as _hf_hub
            _hf_hub.login(token=_hf_token, add_to_git_credential=False)
            print(f"[finetune] HF auth OK (token len={len(_hf_token)})")
        except Exception as _e:
            print(f"[finetune] HF login warning: {_e}")
    else:
        print("[finetune] WARNING: no HF_TOKEN — gated models will fail")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[finetune] Missing dependency: {e}")
        print("Install: pip install transformers peft trl bitsandbytes accelerate")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(__file__))
    from data_loaders import load_dataset_split
    train_data, val_data, _ = load_dataset_split(args.dataset, seed=args.seed)

    if args.split == "train":
        records = train_data
    elif args.split == "val":
        records = val_data
    else:
        raise ValueError(f"Unsupported split for fine-tuning: {args.split}")

    print(f"[finetune] Training on {len(records)} records from {args.dataset}/{args.split}")

    tok_kwargs = {"token": _hf_token} if _hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            **tok_kwargs,
        )
    else:
        _device = "mps" if (hasattr(torch.backends, "mps")
                            and torch.backends.mps.is_available()) else "cpu"
        print(f"[finetune] No CUDA — loading in fp32 on {_device} (no 4-bit quantisation, will be slow)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map=_device,
            trust_remote_code=True,
            **tok_kwargs,
        )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    from datasets import Dataset as HFDataset
    texts  = [record_to_chat(r) for r in records]
    hf_ds  = HFDataset.from_dict({"text": texts})

    os.makedirs(args.output_dir, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=hf_ds,
        peft_config=lora_cfg,
        tokenizer=tokenizer,
    )

    t1_train = time.time()
    trainer.train()
    t2_train = time.time()
    train_time = round(t2_train - t1_train, 1)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    total_time = round(time.time() - t0, 1)
    print(f"\n{'─'*60}")
    print(f"[finetune] Training complete")
    print(f"  Dataset         : {args.dataset}/{args.split} ({len(records)} samples)")
    print(f"  Model           : {args.model_name}")
    print(f"  LoRA r/alpha    : {args.lora_r}/{args.lora_alpha}")
    print(f"  Epochs          : {args.num_epochs}")
    print(f"  Train time      : {train_time:.1f}s  ({train_time/3600:.2f}h)")
    print(f"  Total time      : {total_time:.1f}s")
    print(f"  Adapter saved to: {args.output_dir}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
