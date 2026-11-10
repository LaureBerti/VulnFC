"""
data_loaders.py — Unified dataset loader for VulnFC experiments.

Supported textual datasets (8):
    pubhealth      — HuggingFace "health_fact"
    politifact     — HuggingFace "liar"
    kaggle         — HuggingFace "GonzaloA/fake_news"  (ISOT-style, real/fake news)
    covid19        — HuggingFace "covid_lies"
    climatefever   — HuggingFace "climate_fever" (Diggelmann et al. ACL 2020)
    tiger          — Local JSON: data/tiger/TIGER.json
    snopes         — CSV fallback: data/snopes/snopes.csv
    gossip         — CSV fallback: data/gossip/gossipcop.csv

Supported structured/tabular datasets (4):
    tabfact        — CSV: data/tabfact/tabfact.csv  (Wang et al. 2021; HF script deprecated)
    weather        — CSV fallback: data/weather/weather_claims.csv
    population     — CSV fallback: data/population/population_claims.csv
    flights        — CSV fallback: data/flights/flights_claims.csv

All datasets are binarised (label ∈ {0: FAKE, 1: TRUE}) and split 70/10/20
with a fixed seed (default 42) to match the main paper splits.

Label binarisation for multi-class datasets:
    climatefever / tiger: SUPPORTS→1, REFUTES→0, DISPUTED/NOT_ENOUGH_INFO→dropped
    tabfact: ENTAILED→1, REFUTED→0 (already binary)

CSV format (snopes, gossip, weather, population, flights):
    claim,label      — label must be one of: true/1/yes/real or false/0/no/fake

Synthetic fallback: DISABLED by default since 2026-08-11. If a data file is not found,
this module now RAISES. Generating synthetic data requires an explicit synthetic=True
or VULNFC_ALLOW_SYNTHETIC=1, and prints an unmissable banner when it happens.

A UserWarning is invisible in a batch run, so a silent substitution of synthetic data for real data
can go unnoticed across hundreds of runs. Raising is the only signal that cannot be missed.
"""

import csv
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional



_HF_CONFIGS = {
    "pubhealth": {
        "hf_name":   "health_fact",
        "text_col":  "claim",
        "label_col": "label",
    },
    "politifact": {
        "hf_name":   "liar",
        "text_col":  "statement",
        "label_col": "label",
    },
    "kaggle": {
        "hf_name":   "GonzaloA/fake_news",
        "text_col":  "text",
        "label_col": "label",
    },
    "covid19": {
        "hf_name":   "covid_lies",
        "text_col":  "claim",
        "label_col": "label",
    },
    "climatefever": {
        "hf_name":   "climate_fever",
        "text_col":  "claim",
        "label_col": "claim_label",
        "label_map": {
            0: 1, 1: None, 2: 0, 3: None,
            "SUPPORTS": 1, "DISPUTED": None, "REFUTES": 0, "NOT_ENOUGH_INFO": None,
            "supports": 1, "disputed": None, "refutes": 0, "not_enough_info": None,
        },
    },
}

_CSV_DATASETS = {"snopes", "gossip", "weather", "population", "flights", "tabfact"}

_JSON_DATASETS = {
    "tiger": {
        "path":      "tiger/TIGER.json",
        "text_col":  "claim",
        "label_col": "label",
        "label_map": {
            0: 1, 1: None, 2: 0,
            "SUPPORTS": 1, "DISPUTED": None, "REFUTES": 0,
            "supports": 1, "disputed": None, "refutes": 0,
            "true": 1, "false": 0, "mixture": None,
            "1": 1, "0": 0,
        },
    },
}

_DATA_DIR = Path(__file__).parent.parent / "data"



_LABEL_MAP = {
    0: 1,
    1: 0,
    2: None, 3: None,
    "true":        1, "mostly-true":  1, "half-true":    1,
    "barely-true": 0, "false":        0, "pants-fire":   0,
    "supports":  1, "supported":  1, "entailed":  1,
    "refutes":   0, "refuted":    0, "contradicts": 0,
    "real": 1, "fake": 0,
    "yes":  1, "no":   0,
    "1":    1, "0":    0,
    "not_enough_info": None, "mixture": None, "disputed": None,
}


def _binarise(raw) -> Optional[int]:
    if isinstance(raw, str):
        raw = raw.strip().lower()
    if isinstance(raw, int) and raw not in (0, 1):
        return _LABEL_MAP.get(raw)
    if raw == 1 or raw == "1":
        return 1
    if raw == 0 or raw == "0":
        return 0
    return _LABEL_MAP.get(raw)



def _load_hf(dataset_name: str) -> list:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    cfg = _HF_CONFIGS[dataset_name]

    hf_args = [cfg["hf_name"]]
    if "hf_config" in cfg:
        hf_args.append(cfg["hf_config"])

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*Dataset scripts are no longer supported.*")
        try:
            hf_ds = hf_load(*hf_args)
        except Exception:
            raise

    per_ds_map = cfg.get("label_map")

    def _label(raw):
        if per_ds_map is not None:
            if raw in per_ds_map:
                return per_ds_map[raw]
            if isinstance(raw, str) and raw.lower() in per_ds_map:
                return per_ds_map[raw.lower()]
            return None
        return _binarise(raw)

    records = []
    for split_name in hf_ds:
        for row in hf_ds[split_name]:
            label = _label(row[cfg["label_col"]])
            if label is None:
                continue
            text = str(row[cfg["text_col"]]).strip()
            if not text:
                continue
            records.append({"claim": text, "label": label})
    return records



def _load_csv(dataset_name: str) -> Optional[list]:
    """
    Try to load from data/{dataset_name}/{dataset_name}.csv.
    Expects columns: claim,label  (plus any extras, which are ignored).
    Returns None if file not found.
    """
    _PREFERRED = {"tabfact": f"{dataset_name}_tables.csv"}
    candidates = [
        _DATA_DIR / dataset_name / _PREFERRED.get(dataset_name, f"{dataset_name}.csv"),
        _DATA_DIR / dataset_name / f"{dataset_name}.csv",
        _DATA_DIR / dataset_name / f"{dataset_name}_claims.csv",
        _DATA_DIR / f"{dataset_name}.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None

    if dataset_name in _PREFERRED and csv_path.name != _PREFERRED[dataset_name]:
        warnings.warn(
            f"[data_loaders] '{dataset_name}' is loading {csv_path.name}, which carries no "
            f"evidence column. For TabFact this means the victim sees a statement with no table "
            f"to check it against, and cannot exceed the majority-class baseline. Build the real "
            f"file with:  python data/tabfact/rebuild_with_tables.py",
            UserWarning, stacklevel=2,
        )

    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim = row.get("claim", row.get("text", row.get("statement", ""))).strip()
            if not claim:
                continue
            raw_label = row.get("label", row.get("claim_label", row.get("verdict", "")))
            label = _binarise(raw_label)
            if label is None:
                continue
            rec = {"claim": claim, "label": label}
            ev = (row.get("table_text") or row.get("evidence") or "").strip()
            if ev:
                rec["evidence"] = ev
            sp = (row.get("split") or "").strip().lower()
            if sp:
                rec["split"] = sp
            records.append(rec)
    return records or None



def _load_json(dataset_name: str) -> Optional[list]:
    """
    Load a local JSON or JSON-Lines file for datasets in _JSON_DATASETS.
    Supports both JSON array format ([{...}, ...]) and JSON-Lines ({...}\\n...).
    Returns None if the file is not found.
    """
    import json

    cfg      = _JSON_DATASETS[dataset_name]
    json_path = _DATA_DIR / cfg["path"]
    if not json_path.exists():
        return None

    text_col  = cfg["text_col"]
    label_col = cfg["label_col"]
    label_map = cfg.get("label_map", {})

    def _label(raw):
        if raw in label_map:
            return label_map[raw]
        if isinstance(raw, str) and raw.lower() in label_map:
            return label_map[raw.lower()]
        return _binarise(raw)

    with open(json_path, encoding="utf-8") as f:
        content = f.read().strip()

    try:
        rows = json.loads(content)
        if not isinstance(rows, list):
            rows = [rows]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]

    records = []
    for row in rows:
        claim = str(row.get(text_col, row.get("text", row.get("statement", "")))).strip()
        if not claim:
            continue
        raw_label = row.get(label_col, row.get("verdict", ""))
        label = _label(raw_label)
        if label is None:
            continue
        records.append({"claim": claim, "label": label})
    return records or None



_SYNTHETIC_CLAIMS = [
    ("The global average temperature increased by 1.1°C since pre-industrial times.", 1),
    ("COVID-19 vaccines contain microchips for surveillance.", 0),
    ("The Earth orbits the Sun once every 365.25 days.", 1),
    ("Drinking bleach cures viral infections.", 0),
    ("Sea levels have risen approximately 20 cm since 1900.", 1),
    ("5G towers spread the coronavirus.", 0),
    ("The human body has 206 bones.", 1),
    ("Eating raw garlic cures cancer.", 0),
    ("The Amazon rainforest produces 20 percent of the world's oxygen.", 1),
    ("Bill Gates created COVID-19 in a laboratory.", 0),
    ("Honey never spoils if stored properly.", 1),
    ("Vaccines cause autism.", 0),
    ("The Great Barrier Reef has experienced significant bleaching events.", 1),
    ("The moon landing was staged in a Hollywood studio.", 0),
    ("Regular physical exercise reduces the risk of cardiovascular disease.", 1),
    ("Chemtrails are chemical or biological agents sprayed from aircraft.", 0),
]


def _generate_synthetic(dataset_name: str, n: int = 1000, seed: int = 42) -> list:
    warnings.warn(
        f"[data_loaders] No data file found for '{dataset_name}'. "
        f"Using synthetic dataset ({n} records). "
        f"Place real data at {_DATA_DIR}/{dataset_name}/{dataset_name}.csv "
        f"to use actual experimental results.",
        UserWarning,
        stacklevel=3,
    )
    rng     = random.Random(seed)
    base    = _SYNTHETIC_CLAIMS[:]
    records = []
    for i in range(n):
        claim, label = base[i % len(base)]
        records.append({"claim": claim + f" [v{i//len(base)}]", "label": label})
    rng.shuffle(records)
    return records



def _split(records: list, seed: int) -> tuple:
    """Stratified 70/10/20 split matching the main paper."""
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    def _do(items):
        s = items[:]
        rng.shuffle(s)
        n      = len(s)
        n_tr   = int(0.70 * n)
        n_va   = int(0.10 * n)
        return s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]

    tr_p, va_p, te_p = _do(pos)
    tr_n, va_n, te_n = _do(neg)

    def merge(a, b):
        m = a + b
        rng.shuffle(m)
        return m

    return merge(tr_p, tr_n), merge(va_p, va_n), merge(te_p, te_n)



_ALL_DATASETS = set(_HF_CONFIGS) | _CSV_DATASETS | set(_JSON_DATASETS)


def load_dataset_split(
    dataset_name: str,
    seed:         int  = 42,
    synthetic:    bool = False,
) -> tuple:
    """
    Returns (train, val, test) as lists of {"claim": str, "label": int}.
    Labels: 1 = TRUE / REAL, 0 = FAKE.

    Args:
        dataset_name: one of
          textual HF   — pubhealth, politifact, kaggle, covid19, climatefever
          textual JSON — tiger
          textual CSV  — snopes, gossip
          tabular CSV  — tabfact, weather, population, flights
        seed:         random seed for splits (default 42)
        synthetic:    if True, generate a synthetic fallback when data is missing
    """
    if dataset_name not in _ALL_DATASETS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported: {sorted(_ALL_DATASETS)}"
        )

    records = None

    if dataset_name in _HF_CONFIGS:
        try:
            records = _load_hf(dataset_name)
        except Exception as e:
            _csv_hints = {
                "pubhealth":  "https://github.com/neemakot/Health-Fact-Checking  →  data/pubhealth/pubhealth.csv",
                "politifact": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip  →  data/politifact/politifact.csv",
            }
            hint = _csv_hints.get(dataset_name, "")
            warnings.warn(
                f"[data_loaders] HuggingFace load failed for '{dataset_name}': {e}. "
                f"Trying CSV fallback."
                + (f"  To use real data: {hint}" if hint else ""),
                UserWarning,
                stacklevel=2,
            )

    if records is None and dataset_name in _JSON_DATASETS:
        records = _load_json(dataset_name)
        if records is None:
            warnings.warn(
                f"[data_loaders] JSON file not found for '{dataset_name}'. "
                f"Expected: {_DATA_DIR / _JSON_DATASETS[dataset_name]['path']}. "
                f"Falling back to synthetic.",
                UserWarning,
                stacklevel=2,
            )

    if records is None:
        records = _load_csv(dataset_name)

    if records is None:
        allow_env = os.environ.get("VULNFC_ALLOW_SYNTHETIC", "").lower() in {"1", "true", "yes"}
        if not (synthetic or allow_env):
            raise FileNotFoundError(
                f"No real data found for '{dataset_name}'.\n"
                f"  Expected CSV at: {_DATA_DIR}/{dataset_name}/{dataset_name}.csv\n"
                f"  Fetch it with:   python experiments/fetch_datasets.py --datasets {dataset_name}\n"
                f"  Refusing to fall back to synthetic data. To override for a smoke test only, "
                f"pass synthetic=True or set VULNFC_ALLOW_SYNTHETIC=1 — results so produced "
                f"MUST NOT be reported in the paper."
            )
        print(
            f"\n{'!' * 72}\n"
            f"!! SYNTHETIC DATA IN USE for '{dataset_name}' — NOT REAL RESULTS.\n"
            f"!! Do not report any number derived from this run.\n"
            f"{'!' * 72}\n",
            file=sys.stderr, flush=True,
        )
        records = _generate_synthetic(dataset_name, n=1000, seed=seed)

    if records and all("split" in r for r in records):
        by = {"train": [], "val": [], "test": []}
        for r in records:
            if r["split"] in by:
                by[r["split"]].append(r)
        if all(by.values()):
            train, val, test = by["train"], by["val"], by["test"]
            origin = "official split"
        else:
            train, val, test = _split(records, seed)
            origin = f"random 70/10/20, seed {seed} (official split incomplete)"
    else:
        train, val, test = _split(records, seed)
        origin = f"random 70/10/20, seed {seed}"

    pos_frac = sum(r["label"] for r in test) / max(len(test), 1)
    n_ev = sum(1 for r in test if r.get("evidence"))
    print(
        f"[data] {dataset_name}: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(test_pos_frac={pos_frac:.2f}; {origin}"
        + (f"; evidence on {n_ev}/{len(test)} test items" if n_ev else "") + ")"
    )
    return train, val, test
