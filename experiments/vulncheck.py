#!/usr/bin/env python3
"""vulncheck -- decide whether a poisoning benchmark's own numbers are interpretable.

WHAT THIS IS FOR
----------------
A vulnerability rate (Vulnerability Score, attack success rate, any-flip rate, accuracy drop
under attack) counts CHANGED predictions. A victim whose output does not depend on its input
cannot be changed, so it scores 0.000 -- exactly the score a victim that resists every attack
earns. The two are indistinguishable in the rate itself. Three further failures hide in the same
place: a rate at its ceiling cannot vary, a difference below one claim cannot separate an effect
from one item, and a mean over seeds can conceal an order-of-magnitude spread.

None of those four are visible in the rate. All four are visible in quantities benchmarks already
compute and usually do not publish: the target denominator, the per-seed cells, and the victim's
prediction distribution against a trivial baseline.

So this tool does not load a model, attack anything, or need a GPU. It reads the RESULT FILES a
benchmark already releases and reports, per condition, which of the six checks pass, which fail,
and -- the part that matters for reading someone else's benchmark -- which are NOT CHECKABLE from
what was released. The last category is the finding: in a 23-paper audit the two quantities that
would expose an unmeasurable victim (a trivial baseline, run-to-run spread) were the two most
often missing.

WHAT IT CHECKS
--------------
  D  denominator      n_target_correct present and non-zero; the rate has a population
  G  gate             accuracy clears a majority-class baseline by > min_lift, and no class holds
                      more than max_skew of predictions (needs a QC file; see --qc_csv)
  R  resolution       for two victims compared under one condition, a |difference| no larger than
                      max(1/n1, 1/n2) is attributable to a SINGLE claim and cannot separate an
                      effect from one item at any precision
  S  saturation       vs at the ceiling (or floor): nothing further can register, so a narrow
                      interval there measures determinism, not an effect
  H  homogeneity      every row in a cell comes from one named source; a condition recorded only in
                      a filename cannot be grouped on, so differing stems may pool distinct runs
  V  spread           range across seeds within a cell; a single-seed cell has no spread to report

A condition that passes every applicable check is reported OK. Checks whose inputs are absent are
reported "n/a" and counted separately -- never silently as passes.

USAGE
-----
    python experiments/vulncheck.py                       # audits this project's own outputs
    python experiments/vulncheck.py results_glob='mybench/**/*.csv'
    python experiments/vulncheck.py qc_csv=outputs/c3_sweep/victim_qc.csv
    python experiments/vulncheck.py --write-config        # dump defaults to conf/vulncheck.yaml

Exit status: 0 if no condition fails a check, 1 if any fails, 2 on a usage error. The non-zero
exit is the point -- this is meant to run in a benchmark's CI, before the numbers reach a paper.
"""
from __future__ import annotations

import collections
import csv
import glob
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RATE_COLS = ("vs", "asr", "attack_success_rate", "any_flip_rate", "flip_rate")
DENOM_COLS = ("n_target_correct", "n_correct", "n_eval")
WRONG_DENOM_COLS = ("n_targets", "n_scored", "n_claims")
SEED_COLS = ("seed", "train_seed", "run")
RESULT_COLS = (
    "vs", "ir", "csr", "csr_ir", "asr", "attack_success_rate", "any_flip_rate", "flip_rate",
    "accuracy", "accuracy_clean", "accuracy_poisoned", "accuracy_pre", "accuracy_post",
    "precision", "recall", "f1", "gap", "delta", "diff", "vs_bert", "vs_roberta",
    "acc_bert", "acc_roberta", "vs_delta_from_en",
    "malicious_detected", "malicious_detection_rate", "honest_mean_error",
    "n_target_correct", "n_correct", "n_eval", "n_targets", "n_scored", "n_claims",
    "n_perturbed", "n_candidates", "n_unavailable", "n_injected", "n_poisoned_train",
    "n_sources", "n_honest", "n_malicious", "status", "checkpoint", "verdict", "_file",
    "refusal_rate", "n_target_clean_ok", "n_unavail_adv", "n_unavail_clean", "n_target_adv_ok",
)

OUTCOME_PATTERNS = (r"^time_", r"_s$", r"^finetune_time")
GAP_COLS = ("gap", "delta", "diff")
VICTIM_COLS = ("model", "llm_victim", "victim", "hf_name")


@dataclass
class VulnCheckConfig:
    """All tunables. Thresholds are conventions, not consequences of theory -- the margin is
    reported for every victim so a reader can impose their own."""

    results_glob: str = "outputs/**/*.csv"
    qc_csv: str = ""
    min_lift: float = 0.05
    max_skew: float = 0.90
    ceiling_tol: float = 0.0
    max_report_rows: int = 40
    show_ok: bool = False
    group_by: str = ""
    exclude_patterns: tuple = ("_backup", "INVALID", "SMOKE_ONLY", "_unseeded_", "_preseed_",
                               "_oldmetric_", "_degraded", "_infra_failure", "_prequarantine",
                               "_incomplete_metrics", "_partial_", "_prerequeue")
    filename_keys: tuple = (r"_lr([0-9eE.+-]+)\.csv",)
    seed_patterns: tuple = (r"_seed\d+", r"_s\d+(?=[_.])")


def _first(row: dict, names) -> str | None:
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(pattern: str, exclude: tuple = ()) -> tuple[list[dict], dict]:
    """Read every CSV matching the pattern that carries a rate column.

    Files whose names mark them as superseded (a backup, a smoke test, an unseeded or old-metric
    re-release) are excluded, because a rate in such a file is not a rate the benchmark reports.
    The excluded counts are returned so they can be printed rather than vanishing.
    """
    rows: list[dict] = []
    dropped: dict = {}
    for path in sorted(glob.glob(pattern, recursive=True)):
        name = Path(path).name
        hit = next((e for e in exclude if e in name), None)
        try:
            with open(path, newline="") as fh:
                got = [r for r in csv.DictReader(fh) if _first(r, RATE_COLS) is not None]
        except (OSError, csv.Error) as exc:
            print(f"  ! unreadable: {path} ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        if hit:
            if got:
                dropped[name] = (len(got), hit)
            continue
        for r in got:
            r["_file"] = name
            rows.append(r)
    return rows, dropped


def condition_cols(rows: list[dict], max_levels: int = 25) -> list[str]:
    """Every column that is a CONDITION: a categorical design factor the benchmark varied.

    Conditions are told from outcomes by CARDINALITY rather than by a hardcoded name list. A
    design factor takes a few levels (a budget sweep has eight; a shot mode has three), while an
    outcome or a timing is effectively unique per row. Enumerating names instead would silently
    pool any factor whose name was not anticipated, which is the defect this tool exists to find.
    """
    levels: dict[str, set] = {}
    for r in rows:
        for k, v in r.items():
            if k is None or k in RESULT_COLS or k in SEED_COLS or k in VICTIM_COLS or k == "dataset":
                continue
            if any(re.search(pat, k) for pat in OUTCOME_PATTERNS):
                continue
            if v not in (None, ""):
                levels.setdefault(k, set()).add(v)
    cap = min(max_levels, max(2, len(rows) // 2))
    return sorted(k for k, vs in levels.items() if 1 <= len(vs) <= cap)


def _fn_keys(name: str, pats: tuple) -> tuple:
    """Factors recorded only in a filename (e.g. the learning rate), recovered where possible."""
    out = []
    for pat in pats:
        m = re.search(pat, name)
        out.append(m.group(1) if m else "")
    return tuple(out)


def _fn_stem(name: str, pats: tuple, seed_pats: tuple = ()) -> str:
    """Filename with recovered factors and every seed token removed, for check H."""
    s = name
    for pat in list(pats) + list(seed_pats):
        s = re.sub(pat, "_", s)
    s = re.sub(r"_+", "_", s)
    return re.sub(r"_+(?=\.)", "", s)


def load_qc(path: str) -> dict[tuple[str, str], dict]:
    """Gate inputs keyed by (dataset, victim).

    A QC file usually holds one row per (dataset, victim, training seed), while a result cell is
    keyed by (dataset, victim). Keeping the last row read would report an arbitrary seed's verdict
    as if it were the victim's. We aggregate CONSERVATIVELY instead -- worst lift and worst skew
    across the seeds present -- and report how many seeds that verdict covers, so a victim that is
    degenerate under any seed cannot be waved through by a luckier one.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"  ! qc_csv not found: {p} -- gate will report n/a", file=sys.stderr)
        return {}
    out: dict = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            ds = (r.get("dataset") or "").strip()
            vic = (_first(r, VICTIM_COLS) or "").strip()
            acc, maj = _f(r.get("accuracy")), _f(r.get("majority_baseline"))
            lift = _f(r.get("lift"))
            if lift is None and acc is not None and maj is not None:
                lift = acc - maj
            skew = _f(r.get("skew"))
            rec = out.setdefault((ds, vic), {"lift": lift, "skew": skew, "n_seeds": 0,
                                             "majority": maj})
            if lift is not None:
                rec["lift"] = lift if rec["lift"] is None else min(rec["lift"], lift)
            if skew is not None:
                rec["skew"] = skew if rec["skew"] is None else max(rec["skew"], skew)
            rec["n_seeds"] += 1
    return out


@dataclass
class Cell:
    """One (condition, victim) cell: the seeds that were run for it."""

    dataset: str
    victim: str
    cond: tuple
    rates: list = field(default_factory=list)
    denoms: list = field(default_factory=list)
    wrong_denoms: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    files: list = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.rates) / len(self.rates)

    @property
    def spread(self) -> float:
        return max(self.rates) - min(self.rates)

    @property
    def min_denom(self) -> int | None:
        d = [x for x in self.denoms if x]
        return min(d) if d else None

    @property
    def wrong_denom(self) -> int | None:
        """A target-set size released in place of the correct-set denominator."""
        d = [x for x in self.wrong_denoms if x]
        return min(d) if d else None

    def label(self) -> str:
        c = "/".join(str(x) for x in self.cond if x not in (None, ""))
        return f"{self.dataset}:{self.victim}" + (f" [{c}]" if c else "")


def build_cells(rows: list[dict], cfg) -> tuple[dict, list[str]]:
    """One cell per (corpus, victim, condition), where the condition is EVERY released factor.

    A seed is not a condition -- it is what a cell aggregates over. Everything else that the file
    records and that is not an outcome is part of the cell's identity, so two rows only share a
    cell when the benchmark itself says they were run under the same settings.
    """
    keys = ([k.strip() for k in cfg.group_by.split(",") if k.strip()]
            if cfg.group_by else condition_cols(rows))
    cells: dict = {}
    for r in rows:
        rate = _f(_first(r, RATE_COLS))
        if rate is None:
            continue
        ds = (r.get("dataset") or "?").strip()
        vic = (_first(r, VICTIM_COLS) or "?").strip()
        cond = tuple((r.get(k) or "") for k in keys if k != "dataset")
        cond += _fn_keys(r["_file"], cfg.filename_keys)
        key = (ds, vic, cond)
        cell = cells.setdefault(key, Cell(ds, vic, cond))
        cell.rates.append(rate)
        cell.denoms.append(int(_f(_first(r, DENOM_COLS)) or 0))
        cell.wrong_denoms.append(int(_f(_first(r, WRONG_DENOM_COLS)) or 0))
        cell.seeds.append(_first(r, SEED_COLS))
        cell.files.append(r["_file"])
    return cells, keys


def check_cell(cell: Cell, qc: dict, cfg: VulnCheckConfig) -> list[tuple[str, str, str]]:
    """Return [(status, check, message)] for one cell. status in {FAIL, WARN, n/a, ok}."""
    out = []

    if cell.min_denom:
        out.append(("ok", "D", f"denominator n={cell.min_denom}"))
    elif cell.wrong_denom:
        out.append(("FAIL", "D", f"only the target-set size released (n_targets="
                                 f"{cell.wrong_denom}); the rate is over the correct-set |T+|, "
                                 "which this does not give"))
    else:
        out.append(("FAIL", "D", "no target denominator in the released file: "
                                 "this rate has no stated population"))

    rec = qc.get((cell.dataset, cell.victim))
    if rec is None or rec.get("lift") is None:
        out.append(("n/a", "G", "no trivial baseline released -- a degenerate victim "
                                "would be indistinguishable here"))
    else:
        lift, skew = rec["lift"], rec.get("skew")
        bad = lift <= cfg.min_lift or (skew is not None and skew >= cfg.max_skew)
        why = []
        if lift <= cfg.min_lift:
            why.append(f"lift {lift:+.4f} <= {cfg.min_lift}")
        if skew is not None and skew >= cfg.max_skew:
            why.append(f"skew {skew:.1%} >= {cfg.max_skew:.0%}"
                       + (" (CONSTANT CLASSIFIER)" if skew >= 0.9999 else ""))
        cov = rec.get("n_seeds", 1)
        tail = f" (worst of {cov} seeds)" if cov > 1 else ""
        out.append((("FAIL" if bad else "ok"), "G",
                    ("; ".join(why) if why else
                     f"lift {lift:+.4f}" + (f", skew {skew:.1%}" if skew is not None else ""))
                    + tail))

    hi = [r for r in cell.rates if r >= 1.0 - cfg.ceiling_tol]
    lo = [r for r in cell.rates if r <= 0.0 + cfg.ceiling_tol]
    gate_cleared = bool(rec) and rec.get("lift") is not None and not bad
    if hi:
        out.append(("FAIL", "S", f"rate at the ceiling in {len(hi)}/{len(cell.rates)} seed(s): "
                                 "nothing further can register, so a narrow interval here "
                                 "measures determinism, not an effect"))
    elif lo and gate_cleared:
        out.append(("ok", "S", f"rate is 0.000 in {len(lo)}/{len(cell.rates)} seed(s), but the "
                               "victim cleared the gate: this reads as resistance"))
    elif lo:
        n = cell.min_denom
        out.append(("FAIL", "S", f"rate is exactly 0.000 in {len(lo)}/{len(cell.rates)} seed(s)"
                                 + (f" over n={n}" if n else "")
                                 + ", with no gate to tell resistance from a victim that "
                                   "cannot be flipped"))
    else:
        out.append(("ok", "S", f"rate {cell.mean:.4f} off both bounds"))

    stems = {_fn_stem(f, cfg.filename_keys, cfg.seed_patterns) for f in cell.files}
    if len(stems) > 1:
        out.append(("FAIL", "H", f"rows come from {len(stems)} differently-named sources after "
                                 "removing seed and recovered factors: a condition may be recorded "
                                 "only in filenames, so this cell may pool distinct experiments"))
    else:
        out.append(("ok", "H", f"one condition, {len(cell.rates)} run(s)"))

    if len(cell.rates) < 2:
        out.append(("n/a", "V", "single seed: no run-to-run spread to report"))
    elif cell.spread > abs(cell.mean) and cell.mean != 0:
        out.append(("WARN", "V", f"spread {cell.spread:.4f} exceeds the mean {cell.mean:.4f} "
                                 f"over {len(cell.rates)} seeds "
                                 f"[{min(cell.rates):.4f}, {max(cell.rates):.4f}]"))
    else:
        out.append(("ok", "V", f"spread {cell.spread:.4f} over {len(cell.rates)} seeds"))

    return out


def check_resolution(cells: dict, cfg: VulnCheckConfig) -> list[str]:
    """R -- for victims compared under one condition, is the difference bigger than one claim?"""
    findings = []
    by_cond: dict = collections.defaultdict(list)
    for (ds, vic, cond), cell in cells.items():
        by_cond[(ds, cond)].append(cell)
    for (ds, cond), group in sorted(by_cond.items(), key=lambda kv: str(kv[0])):
        group = [c for c in group if c.min_denom]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                diff = abs(a.mean - b.mean)
                bound = max(1.0 / a.min_denom, 1.0 / b.min_denom)
                if diff <= bound:
                    claims = diff * min(a.min_denom, b.min_denom)
                    findings.append(
                        f"FAIL R  {a.label()} vs {b.victim}: |delta|={diff:.4f} <= "
                        f"max(1/{a.min_denom},1/{b.min_denom})={bound:.4f} "
                        f"-- {claims:.1f} claim(s) wide; not separable from a single item"
                    )
    return findings


def check_gap_stability(pattern: str, cfg: VulnCheckConfig) -> list[str]:
    """P -- is a released between-victim difference stable across the seeds behind it?

    A benchmark that reports one architecture gap per (corpus, attack, learning rate) has usually
    averaged several training seeds into it, or run one. Either way the quantity it claims is the
    difference, so the difference is what has to be shown stable. A cell-by-cell spread check
    misses this: each victim's own rate can look settled while the gap between them moves by an
    order of magnitude, which is how a single lucky seed becomes a published architecture effect.
    """
    groups: dict = collections.defaultdict(list)
    for path in sorted(glob.glob(pattern, recursive=True)):
        if any(e in Path(path).name for e in cfg.exclude_patterns):
            continue
        try:
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    g = _f(_first(r, GAP_COLS))
                    if g is None:
                        continue
                    key = (r.get("dataset") or "?", r.get("attack") or r.get("attack_type") or "?",
                           r.get("learning_rate") or "")
                    groups[key].append((_first(r, SEED_COLS), g))
        except (OSError, csv.Error):
            continue

    findings = []
    for key, pairs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(pairs) < 2:
            continue
        gaps = [g for _, g in pairs]
        rng = max(gaps) - min(gaps)
        mean = sum(gaps) / len(gaps)
        signs = {g > 0 for g in gaps if g != 0}
        ds, atk, lr = key
        tag = f"{ds}:{atk}" + (f"/lr{lr}" if lr else "")
        vals = ", ".join(f"{g:+.4f}" for g in gaps)
        if len(signs) > 1:
            findings.append(f"FAIL P  {tag}: the released gap changes SIGN across "
                            f"{len(gaps)} seeds [{vals}] -- the direction of the effect is "
                            f"not determined by the design")
        elif abs(mean) > 0 and rng > abs(mean):
            findings.append(f"FAIL P  {tag}: the released gap ranges {rng:.4f} across "
                            f"{len(gaps)} seeds [{vals}] about a mean of {mean:+.4f} -- a single "
                            f"seed moves it by more than its own size")
    return findings


def report(cfg: VulnCheckConfig) -> int:
    pattern = cfg.results_glob
    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)
    rows, dropped = load_rows(pattern, cfg.exclude_patterns)
    if not rows:
        print(f"vulncheck: no CSV with a rate column matched {pattern}", file=sys.stderr)
        return 2
    qc = load_qc(cfg.qc_csv)
    cells, keys = build_cells(rows, cfg)

    print("=" * 78)
    print("vulncheck -- is this benchmark's vulnerability rate interpretable?")
    print("=" * 78)
    n_drop = sum(c for c, _ in dropped.values())
    print(f"  rate rows read : {len(rows)}"
          + (f"   ({n_drop} excluded as superseded)" if n_drop else ""))
    if dropped:
        for nm, (c, why) in sorted(dropped.items(), key=lambda kv: -kv[1][0])[:6]:
            print(f"      -{c:5}  {nm}  [{why}]")
    print(f"  cells          : {len(cells)}  (condition x victim)")
    print(f"  grouped on     : {', '.join(keys) or '(none found)'}"
          + (" + factors recovered from filenames" if cfg.filename_keys else ""))
    print(f"  gate inputs    : {len(qc) or 'none supplied'}")
    print(f"  thresholds     : lift > {cfg.min_lift}, skew < {cfg.max_skew:.0%} "
          f"(conventions; margins reported)")
    print()

    tally = collections.Counter()
    shown = 0
    for key in sorted(cells, key=lambda k: (k[0], k[1], str(k[2]))):
        cell = cells[key]
        res = check_cell(cell, qc, cfg)
        for status, _, _ in res:
            tally[status] += 1
        worst = ("FAIL" if any(s == "FAIL" for s, _, _ in res)
                 else "WARN" if any(s == "WARN" for s, _, _ in res) else "ok")
        tally[f"cell:{worst}"] += 1
        if worst == "ok" and not cfg.show_ok:
            continue
        if shown >= cfg.max_report_rows:
            continue
        shown += 1
        print(f"[{worst:4}] {cell.label()}   seeds={len(cell.rates)} "
              f"rate={cell.mean:.4f}")
        for status, check, msg in res:
            if status == "ok" and not cfg.show_ok:
                continue
            print(f"         {status:4} {check}  {msg}")
        print()

    ungated = [(k, v) for k, v in sorted(qc.items())
               if not any(c.dataset == k[0] and c.victim == k[1] for c in cells.values())]
    if ungated:
        print("-" * 78)
        print("PRE-ATTACK GATE -- victims with no attack results yet")
        print("-" * 78)
        for (ds, vic), rec in ungated:
            lift, skew = rec.get("lift"), rec.get("skew")
            if lift is None:
                continue
            bad = lift <= cfg.min_lift or (skew is not None and skew >= cfg.max_skew)
            note = (" CONSTANT CLASSIFIER" if skew is not None and skew >= 0.9999
                    else " below the trivial predictor" if lift < 0 else "")
            skew_s = f" skew={skew:.1%}" if skew is not None else ""
            print(f"  {'FAIL' if bad else 'ok  '} {ds}:{vic}  lift={lift:+.4f}{skew_s}"
                  f"{note}  (worst of {rec.get('n_seeds', 1)} seeds)")
            tally["gate:FAIL" if bad else "gate:ok"] += 1

    gap_findings = check_gap_stability(pattern, cfg)
    if gap_findings:
        print("-" * 78)
        print("GAP STABILITY -- released differences across the seeds behind them")
        print("-" * 78)
        for f in gap_findings[: cfg.max_report_rows]:
            print("  " + f)
        print()

    res_findings = check_resolution(cells, cfg)
    if res_findings:
        print("-" * 78)
        print("RESOLUTION -- differences no wider than one claim")
        print("-" * 78)
        for f in res_findings[: cfg.max_report_rows]:
            print("  " + f)
        if len(res_findings) > cfg.max_report_rows:
            print(f"  ... {len(res_findings) - cfg.max_report_rows} more")
        print()

    n_fail = tally["cell:FAIL"]
    if tally["gate:FAIL"] or tally["gate:ok"]:
        print(f"  pre-attack gate      {tally['gate:ok']} pass, "
              f"{tally['gate:FAIL']} FAIL (no attack run yet)")
    if gap_findings:
        print(f"  unstable released differences                  {len(gap_findings)}")
    print("=" * 78)
    print(f"  cells OK            {tally['cell:ok']}")
    print(f"  cells with WARNING  {tally['cell:WARN']}")
    print(f"  cells FAILING       {n_fail}")
    print(f"  checks not checkable from the released files   {tally['n/a']}")
    print(f"  cross-victim differences inside one claim      {len(res_findings)}")
    print("=" * 78)
    if n_fail or res_findings or gap_findings or tally["gate:FAIL"]:
        print("A failing cell does not mean the attack is weak. It means the number cannot be")
        print("read as evidence of robustness, because the check that would separate the two")
        print("was not satisfiable. Fix the measurement before interpreting the rate.")
    return 1 if (n_fail or res_findings or gap_findings or tally["gate:FAIL"]) else 0


def _override_from_argv(argv: list[str]) -> VulnCheckConfig:
    """Apply `key=value` overrides onto the default config using the standard library only.

    This is the fallback path when Hydra is not installed. It exists because the tool's whole
    premise is that a reader can point it at someone else's benchmark and run it in CI: a headline
    command that dies with ModuleNotFoundError on a stock interpreter defeats that, and the paper
    states the tool has no third-party dependency. Hydra is still used when present, so the
    documented `--write-config` and composition behaviour are unchanged for anyone who has it.
    """
    cfg = VulnCheckConfig()
    for arg in argv:
        if arg.startswith("-") or "=" not in arg:
            continue
        key, _, raw = arg.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown option {key!r}; known: {', '.join(vars(cfg))}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            val = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, tuple):
            val = tuple(x for x in raw.split(",") if x)
        else:
            val = raw
        setattr(cfg, key, val)
    return cfg


def main() -> int:
    if "--write-config" in sys.argv:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise SystemExit("--write-config needs omegaconf (pip install omegaconf); "
                             "the checks themselves do not.")
        out = ROOT / "conf" / "vulncheck.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(VulnCheckConfig), out)
        print(f"wrote {out}")
        return 0

    try:
        import hydra
        from hydra.core.config_store import ConfigStore
    except ImportError:
        return report(_override_from_argv(sys.argv[1:]))

    cs = ConfigStore.instance()
    cs.store(name="vulncheck_config", node=VulnCheckConfig)

    rc: dict = {}

    @hydra.main(version_base=None, config_name="vulncheck_config")
    def _run(cfg) -> None:
        rc["code"] = report(cfg)

    _run()
    return rc.get("code", 2)


if __name__ == "__main__":
    sys.exit(main())
