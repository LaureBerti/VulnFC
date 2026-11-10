#!/usr/bin/env python3
"""
fetch_datasets.py — materialise REAL datasets to local CSV, with provenance.

Two properties this enforces:
  1. Every dataset is materialised ONCE to a local CSV, and all later runs read that CSV.
     `datasets >= 4.0` removed script-based loaders, so resolving a corpus from the Hub at
     experiment time is version-dependent; reading a local CSV is deterministic. Note that
     `data_loaders.load_dataset_split()` falls back to `_generate_synthetic()` for a missing
     dataset, warning but not failing, so materialising the real data first is what keeps a
     synthetic placeholder out of a run.
  2. Provenance travels with the data. Every fetch records source, retrieval date,
     row count, label distribution and a SHA-256 in data/MANIFEST.json.

Output: data/<name>/<name>.csv with columns (claim, label); label 1 = TRUE/REAL, 0 = FAKE.

Compute: CPU / network only.

Usage
-----
    python fetch_datasets.py --all
    python fetch_datasets.py --datasets kaggle covid19
    python fetch_datasets.py --all --dry_run     # report plan, download nothing
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"



SOURCES = {
    "kaggle": {
        "hf":        "GonzaloA/fake_news",
        "text_col":  "text",
        "label_col": "label",
        "label_map": None,
        "note":      "Primary textual corpus.",
    },
    "covid19": {
        "hf":        "nanyy1025/covid_fake_news",
        "text_col":  "tweet",
        "label_col": "label",
        "label_map": "auto",
        "substitute_for": "covid_lies",
        "note":      "covid_lies was DELETED from the HuggingFace Hub (RepositoryNotFoundError), "
                     "which is why covid19 fell back to synthetic in 30/30 runs. This is a "
                     "DIFFERENT corpus and must be renamed in the paper, not passed off as covid_lies.",
    },
    "pubhealth": {
        "hf":        "Jezzarax/pubhealth-converted",
        "config":    "pubhealth_source",
        "text_col":  "claim",
        "label_col": "label",
        "label_map": {"true": 1, "false": 0, "mixture": None, "unproven": None},
        "note":      "PUBHEALTH (Kotonya & Toni, EMNLP 2020). The 'health_fact' and "
                     "ImperialCollegeLondon/health_fact ids are script-based and fail on "
                     "datasets>=4.0; this is a parquet conversion. 4-way labels reduced to "
                     "binary by dropping mixture/unproven.",
    },
}

UNRECOVERABLE = {
    "gossip":     "GossipCop (FakeNewsNet, Shu et al.). Real in only 10/40 runs; the local "
                  "gossipcop.csv is gone. Candidate mirrors exist but none is the same corpus: "
                  "LittleFish-Coder/Fake_News_GossipCop ships ~800MB of precomputed embeddings and "
                  "stalls; Jinyan1/GossipCop splits are MF/HF/MR/HR (machine- vs human-generated), "
                  "i.e. a machine-text-detection variant, NOT the original fake/real corpus. "
                  "Substituting it would silently change what the dataset measures — needs a decision.",
    "snopes":     "CSV-only, no source documented anywhere in the repo. Real in 10/38 runs; "
                  "the CSV that produced those is gone. No public HF mirror found.",
    "weather":    "Truth-discovery style tabular dataset. NEVER real (0/51 runs). No source documented.",
    "population": "Truth-discovery style tabular dataset. NEVER real (0/46 runs). No source documented.",
    "flights":    "Truth-discovery style tabular dataset. NEVER real (0/44 runs). No source documented. "
                  "The classic Flight benchmark (Li et al., VLDB 2013) is a candidate replacement "
                  "but is a DIFFERENT dataset and needs an explicit decision.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--min_rows", type=int, default=200,
                   help="Fail a dataset that yields fewer rows than this")
    return p.parse_args()


def _binarise(raw, mapping):
    """Map an upstream label to our convention: 1 = TRUE/REAL, 0 = FAKE. None = drop."""
    if mapping:
        return mapping.get(raw, mapping.get(str(raw).lower()))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw if raw in (0, 1) else None
    s = str(raw).strip().lower()
    if s in {"true", "real", "1", "supports", "entailed"}:
        return 1
    if s in {"fake", "false", "0", "refutes", "refuted"}:
        return 0
    return None


def _infer_mapping(values) -> dict:
    """Infer a label mapping from observed values, conservatively.

    Only handles the unambiguous cases. Anything else raises, because guessing a
    label polarity silently inverts every downstream result.
    """
    uniq = sorted({str(v).strip().lower() for v in values})
    known = {
        "0": 0, "1": 1, "true": 1, "false": 0, "real": 1, "fake": 0,
        "supports": 1, "refutes": 0, "entailed": 1, "refuted": 0,
    }
    if all(u in known for u in uniq):
        return {u: known[u] for u in uniq}
    raise ValueError(
        f"Cannot infer label polarity from values {uniq}. "
        f"Add an explicit label_map for this dataset rather than guessing."
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(name: str, cfg: dict, min_rows: int) -> dict:
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset(cfg["hf"], cfg["config"]) if cfg.get("config") else load_dataset(cfg["hf"])

    rows, dropped = [], 0
    raw_labels = []
    for split in ds:
        for rec in ds[split]:
            raw_labels.append(rec[cfg["label_col"]])
    mapping = cfg.get("label_map")
    if mapping == "auto":
        mapping = _infer_mapping(raw_labels)

    for split in ds:
        for rec in ds[split]:
            text = (rec.get(cfg["text_col"]) or "").strip()
            lab = _binarise(rec[cfg["label_col"]], mapping)
            if not text or lab is None:
                dropped += 1
                continue
            rows.append((text, lab))

    if len(rows) < min_rows:
        raise RuntimeError(f"only {len(rows)} usable rows (min {min_rows})")

    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{name}.csv"
    if out_csv.exists():
        backup = out_dir / f"{name}_previous_{date.today().isoformat()}.csv"
        out_csv.replace(backup)
        print(f"    existing CSV preserved -> {backup.name}", file=sys.stderr)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim", "label"])
        w.writerows(rows)

    n_pos = sum(l for _, l in rows)
    return {
        "dataset":        name,
        "source":         cfg["hf"],
        "source_type":    "huggingface",
        "retrieved":      date.today().isoformat(),
        "n_rows":         len(rows),
        "n_pos":          n_pos,
        "n_neg":          len(rows) - n_pos,
        "pos_frac":       round(n_pos / len(rows), 4),
        "dropped":        dropped,
        "label_mapping":  {str(k): v for k, v in (mapping or {}).items()} or "identity",
        "sha256":         sha256(out_csv),
        "path":           str(out_csv.relative_to(DATA_DIR.parent)),
        "substitute_for": cfg.get("substitute_for"),
        "note":           cfg["note"],
        "fetch_seconds":  round(time.time() - t0, 1),
    }


def main():
    args = parse_args()
    names = list(SOURCES) if (args.all or not args.datasets) else args.datasets

    print(f"[fetch] targets: {', '.join(names)}", file=sys.stderr)
    if args.dry_run:
        for n in names:
            c = SOURCES.get(n)
            print(f"  {n:12} <- {c['hf'] if c else 'UNKNOWN'}", file=sys.stderr)
        print(f"\n  unrecoverable (no public source): {', '.join(UNRECOVERABLE)}", file=sys.stderr)
        return

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    ok, failed = [], []
    for n in names:
        cfg = SOURCES.get(n)
        if not cfg:
            print(f"[skip] {n}: not in SOURCES registry", file=sys.stderr)
            failed.append((n, "not in registry"))
            continue
        print(f"[fetch] {n} <- {cfg['hf']} ...", file=sys.stderr)
        try:
            entry = fetch_one(n, cfg, args.min_rows)
            manifest[n] = entry
            ok.append(n)
            sub = f"  SUBSTITUTE for {entry['substitute_for']}" if entry["substitute_for"] else ""
            print(f"  ✓ {entry['n_rows']} rows (pos_frac={entry['pos_frac']}) "
                  f"in {entry['fetch_seconds']}s{sub}", file=sys.stderr)
        except Exception as exc:
            print(f"  ✗ {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
            failed.append((n, f"{type(exc).__name__}: {str(exc)[:160]}"))

    manifest["_unrecoverable"] = UNRECOVERABLE
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\n[done] fetched: {', '.join(ok) if ok else '(none)'}", file=sys.stderr)
    if failed:
        print("[failed]", file=sys.stderr)
        for n, why in failed:
            print(f"  - {n}: {why}", file=sys.stderr)
    print(f"[manifest] {MANIFEST}", file=sys.stderr)
    print("\nDatasets with no public source (must be decided, NOT silently substituted):",
          file=sys.stderr)
    for n, why in UNRECOVERABLE.items():
        print(f"  - {n}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
