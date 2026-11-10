#!/usr/bin/env python3
"""Every statistic quoted in the victim-scale section (C2), computed from the three-seed grid.

The section was written when only seed 42 had finished, and its numbers stayed at those values
after the other two seeds landed. Rather than re-typing them, this script is the single source:
each number the prose quotes is printed here with the same label, so the two can be diffed.

Design notes that change the answers, and are therefore not incidental:

  * Cells are seed-averaged before any test. VS at (victim, corpus, attack) is measured three times;
    treating those as three independent observations is pseudo-replication and inflates every
    p-value's apparent strength.
  * Page's test is built around a predicted ordering. The pre-registration fixed the ordering
    (VS against ascending victim scale) without fixing a sign, so the p-value is reported
    two-sided.
  * The partial correlations are the point of the section. Scale and target accuracy are collinear
    here (larger victims are more accurate), so the marginal correlation cannot separate them; only
    the partials can, and they are reported whichever way they fall.

Usage:  python experiments/stats_scale.py
"""
from __future__ import annotations

import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats as sp

from make_tables import rows, _page_trend

VICTIMS = [("llama3.2:latest", "Llama-3.2", 3.2),
           ("phi3.5:latest", "Phi-3.5", 3.8),
           ("qwen2.5:14b", "Qwen-2.5", 14.8)]
CORPORA = ["kaggle", "pubhealth", "covid19"]
ATTACKS = ["llm_generated", "semantic_paraphrase", "prompt_injection"]


def partial(x, y, z):
    """Correlation of x and y with z partialled out, via residuals of the linear fits."""
    x, y, z = map(np.asarray, (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r, p = sp.pearsonr(rx, ry)
    return r, p, len(x) - 3


def main() -> int:
    rs = rows("c1_v3/c1_v3_results.csv")
    cell = defaultdict(list)
    acc = defaultdict(list)
    for r in rs:
        k = (r["llm_victim"], r["dataset"], r["attack_type"])
        cell[k].append(float(r["vs"]))
        acc[k].append(int(r["n_target_correct"]) / int(r["n_targets"]))

    seeds = sorted({r["seed"] for r in rs}, key=int)
    print(f"seeds present: {', '.join(seeds)}   conditions: {len(rs)}\n")

    print("MEAN VS BY SCALE")
    means = {}
    for k, disp, b in VICTIMS:
        v = [x for c in CORPORA for a in ATTACKS for x in cell[(k, c, a)]]
        means[k] = st.mean(v)
        ta = [x for c in CORPORA for a in ATTACKS for x in acc[(k, c, a)]]
        print(f"  {disp:12} {b:5.1f}B  mean VS {means[k]:.3f}   target acc {st.mean(ta):.3f}")
    ordered = [means[k] for k, _, _ in VICTIMS]
    print(f"  monotone in mean VS? {'YES' if ordered == sorted(ordered) else 'NO'}"
          f"  ({' -> '.join(f'{m:.3f}' for m in ordered)})\n")

    blocks = [[st.mean(cell[(k, c, a)]) for k, _, _ in VICTIMS]
              for c in CORPORA for a in ATTACKS]
    L, z, p = _page_trend(blocks)
    wins = sum(1 for b in blocks if b[-1] > b[0])
    mono = sum(1 for b in blocks if b[0] <= b[1] <= b[2])
    print("PAGE'S TREND TEST (seed-averaged cells, 9 blocks)")
    print(f"  L = {L:.0f} vs {len(blocks)*3*16/4:.0f} expected, z = {z:+.2f}, two-sided p = {p:.4f}")
    print(f"  largest > smallest in {wins}/{len(blocks)} blocks")
    print(f"  fully ordered 3.2<=3.8<=14.8 in {mono}/{len(blocks)} blocks\n")

    lo = [st.mean(cell[(VICTIMS[0][0], c, a)]) for c in CORPORA for a in ATTACKS]
    hi = [st.mean(cell[(VICTIMS[2][0], c, a)]) for c in CORPORA for a in ATTACKS]
    d = [h - l for h, l in zip(hi, lo)]
    W, pw = sp.wilcoxon(hi, lo)
    tcrit = sp.t.ppf(0.975, len(d) - 1)
    half = tcrit * st.stdev(d) / len(d) ** 0.5
    print("LARGEST (14.8B) vs SMALLEST (3.2B)")
    print(f"  mean difference {st.mean(d):+.3f}  95% CI [{st.mean(d)-half:+.3f}, {st.mean(d)+half:+.3f}]")
    print(f"  Wilcoxon signed-rank W = {W:.0f}, p = {pw:.4f}\n")

    xs = [math.log(b) for k, _, b in VICTIMS for c in CORPORA for a in ATTACKS]
    ys = [st.mean(cell[(k, c, a)]) for k, _, b in VICTIMS for c in CORPORA for a in ATTACKS]
    ta = [st.mean(acc[(k, c, a)]) for k, _, b in VICTIMS for c in CORPORA for a in ATTACKS]
    rho, prho = sp.spearmanr(xs, ys)
    print(f"SPEARMAN log-params vs VS (n={len(xs)} seed-averaged cells)")
    print(f"  rho = {rho:+.3f}, p = {prho:.4f}")
    rho2, prho2 = sp.spearmanr(xs, ta)
    print(f"  collinearity check, log-params vs target accuracy: rho = {rho2:+.3f}, p = {prho2:.4f}\n")

    r1, p1, dof = partial(xs, ys, ta)
    r2, p2, _ = partial(ta, ys, xs)
    print(f"PARTIAL CORRELATIONS (n={len(xs)}, dof={dof})")
    print(f"  VS ~ scale, controlling target accuracy    rho = {r1:+.3f}  p = {p1:.4f}")
    print(f"  VS ~ target accuracy, controlling scale    rho = {r2:+.3f}  p = {p2:.4f}")
    sep = (p1 < 0.05) != (p2 < 0.05)
    print(f"  separable at alpha=0.05? {'YES' if sep else 'NO — both on the same side of 0.05'}")
    if not sep:
        print("  => the section may NOT claim the scale effect survives its control while the")
        print("     capability explanation does not. With three victims the two are collinear")
        print("     and this design cannot separate them.\n")

    pool = [(int(r["n_target_correct"]) / int(r["n_targets"]), float(r["vs"]))
            for r in rs if r.get("n_targets") and int(r["n_targets"])]
    rp, pp = sp.spearmanr([a for a, _ in pool], [v for _, v in pool])
    print(f"POOLED over {len(pool)} C1 conditions recording both quantities")
    print(f"  target accuracy vs VS: rho = {rp:+.3f}, p = {pp:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
