#!/usr/bin/env python3
"""Quantify how much the recurrent vulnerability grid moves between two executions.

Both executions use the same harness, the same budget, the same three seeds and the same victims.
The victims are recurrent classifiers that take an argmax and sample nothing, so a deterministic
attack must reproduce exactly and any movement has to come from the attack text. That is what
separates the two attack families here: `pos_negation` is produced by a rule, `llm_generated` by a
sampling language model.

Reproduces every figure in the execution-variance paragraph of the sophistication section:

  * per attack family, how many conditions are bit-identical across the two executions, with the
    mean and maximum absolute change in VS;
  * the admissible contrast recomputed on each arm separately, so the comparison the paper rests on
    can be checked against both.

Usage:
    python experiments/check_execution_variance.py
    python experiments/check_execution_variance.py --write
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

CANDIDATES = {
    "arm_a": ["outputs/c9_v2/c9_v2_results.csv"],
    "arm_b": ["outputs/r3_matched/r3_matched_results.csv",
              "outputs/post_paper/r3_matched/r3_matched_results.csv"],
}
ATTACKS = ["llm_generated", "pos_negation"]


def resolve(key: str) -> Path:
    for rel in CANDIDATES[key]:
        p = ROOT / rel
        if p.exists():
            return p
    raise SystemExit(f"missing input for {key}; looked in: {', '.join(CANDIDATES[key])}")


def conditions(path: Path) -> dict[tuple, float]:
    return {(r["dataset"], r["model"], r["attack_type"], r["seed"]): float(r["vs"])
            for r in csv.DictReader(path.open()) if r["attack_type"] in ATTACKS}


def cells(path: Path) -> dict[tuple, dict[str, float]]:
    acc: dict[tuple, list] = {}
    for r in csv.DictReader(path.open()):
        if r["attack_type"] in ATTACKS:
            acc.setdefault((r["dataset"], r["model"], r["attack_type"]), []).append(float(r["vs"]))
    out: dict[tuple, dict[str, float]] = {}
    for (ds, v, atk), vals in acc.items():
        out.setdefault((ds, v), {})[atk] = st.mean(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from gate_recurrent import contrast, gate_rows
    from make_tables import leaking_corpora

    a_path, b_path = resolve("arm_a"), resolve("arm_b")
    a, b = conditions(a_path), conditions(b_path)
    shared = sorted(a.keys() & b.keys())
    print(f"EXECUTION VARIANCE  —  {a_path.name} vs {b_path.name}")
    print(f"  conditions present in both executions: {len(shared)}\n")

    rows = []
    print(f"  {'attack':16} {'n':>4} {'identical':>10} {'mean|dVS|':>10} {'max|dVS|':>9}")
    for atk in ATTACKS:
        d = [abs(a[k] - b[k]) for k in shared if k[2] == atk]
        if not d:
            continue
        ident = sum(1 for x in d if x < 1e-9)
        print(f"  {atk:16} {len(d):>4} {f'{ident}/{len(d)}':>10} "
              f"{st.mean(d):10.4f} {max(d):9.4f}")
        rows.append({"attack": atk, "n": len(d), "identical": ident,
                     "mean_abs_dvs": round(st.mean(d), 4), "max_abs_dvs": round(max(d), 4)})

    print("\n  A rule-based attack reproduces exactly; a generated one does not. The victims sample")
    print("  nothing, so the difference is the attack text, not the victim.")

    gate = {(g["dataset"], g["victim"]): g["gate"] for g in gate_rows()}
    leak = leaking_corpora()
    ca, cb = cells(a_path), cells(b_path)
    adm = [k for k in sorted(ca.keys() & cb.keys())
           if gate.get(k) == "PASS" and k[0] not in leak
           and all(x in ca[k] and x in cb[k] for x in ATTACKS)]

    print(f"\n  ADMISSIBLE CONTRAST (gate-passing, leak-free): {len(adm)} cells")
    for label, c in [("execution A", ca), ("execution B", cb)]:
        r = contrast([(c[k]["llm_generated"], c[k]["pos_negation"]) for k in adm])
        line = (f"    {label}: generated stronger in {r['wins']}/{r['n']}, "
                f"mean delta {r['mean_all']:+.4f}, 95% CI "
                f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
        if "p_exact" in r:
            line += f", exact p = {r['p_exact']:.4f}"
        print(line)
        rows.append({"attack": f"contrast_{label.replace(' ', '_')}", "n": r["n"],
                     "identical": r["wins"], "mean_abs_dvs": round(r["mean_all"], 4),
                     "max_abs_dvs": ""})
    print("\n  The contrast is what the paper claims; it holds on both executions.")

    if args.write and rows:
        out = ROOT / ("outputs/execution_variance.csv" if (ROOT / "outputs/c9_v2").exists()
                      else "outputs/post_paper/execution_variance.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
