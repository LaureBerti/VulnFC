# VulnFC — Truth under Input Poisoning Attacks

Code and reproduction guide for the paper:

> **Truth under Input Poisoning Attacks: Benchmarking Fact-Checking Architectures**

---

## Overview

VulnFC evaluates five attack families against six deep learning fact-checking architectures across twelve datasets (eight textual, four structured).

| Claim | Finding |
|-------|---------|
| C1 — LLM victim vulnerability | Llama-3.2 reaches VS ≥ 0.20 on all 6 textual datasets |
| C2 — SSTF-VI robustness | Accuracy ≥ 0.85 under Sybil injection up to MSR = 0.20 |
| C3 — White-box upper bound | FGSM achieves VS = 0.40–0.87 at ε = 0.30 |
| C4 — SLM structured vulnerability | BERT reaches VS = 0.23–0.67 on structured datasets |
| C5 — Architecture-dependent susceptibility | LLM-generated attacks advantage LSTM-title in 6/9 datasets |

The five findings above are stated in Vulnerability Score as originally measured. Read them with
the section immediately below: on the label-changing attack families that scoring is not
interpretable, and the re-measured values ship alongside the originals (`*_dfr.csv` per track,
`common_grid/` for the encoder cells, `common_grid_published_vs/` for the originals).

---

## Attack label semantics (read before interpreting any score)

An attack either preserves the truth value of the claim it rewrites, or changes it. The two cannot
be scored against the same reference label.

| | Attacks | Metric | Scored against |
|---|---|---|---|
| Label-preserving | `semantic_paraphrase`, `prompt_injection`, `fgsm`, `pgd` | Vulnerability Score (VS) | the original label |
| Label-changing | `llm_generated`, `semantic_negation`, `pos_negation`, `content_perturbation`, `sybil` | Detection Failure Rate (DFR) | the complement of the original label |

A label-changing attack rewrites "X is true" into a claim that is false. A victim that still
answers with the original label is now wrong, but VS scored against that original label counts the
answer as a success for the victim. The consequence is measurable: a perfect fact-checker scores
VS = 1.000 on every label-changing family, and DFR = 0.000.

`experiments/attack_semantics.py` is the single source of this partition. It raises
`UndeclaredAttack` rather than guessing, so a new attack family cannot be scored until its label
semantics are declared. Every runner imports it; none of them chooses a metric locally.

On label-changing attacks the two quantities are definitionally complementary: scoring one fixed
set of predictions against the original label and against its complement must give
`DFR = 1 - VS`, because every prediction counted as a success by one is counted as a failure by
the other.

**This is not a check on the re-measurement, and must not be used as one.** It holds only for a
re-scoring of fixed predictions. The re-measurement is not that: encoder and recurrent victims are
retrained (checkpoints are not redistributable), and the attacks are regenerated, which is
non-deterministic for every LLM-generated family. Both the predictions and the scored item set
therefore differ from the published run. Measured per condition, the identity fails on 35 of 36
encoder cells, 17 of 18 recurrent cells, and 26 of 27 LLM cells, with deviations up to 0.18 --
which is the expected consequence of a fresh measurement, not evidence of an error in either run.
Aggregate means can still agree closely while nearly every cell disagrees, so an agreement of
means is not a verification either.

What this means for reading the numbers: a re-measured condition is a new measurement of the same
condition, not an arithmetic transformation of the old one, and the two cannot be reconciled cell
by cell.

**Re-measurement status.** The LLM-victim track, the fully crossed encoder grid and the
recurrent-victim arm are all re-measured under the corrected semantics. The recurrent
sophistication grid is converted by exact re-scoring instead
(`experiments/rescore_sophistication.py`), since its victims are deterministic and binary. Result
files carry a `metric` column naming the quantity in `metric_value`, plus a `label_semantics`
column, so no row is ambiguous about what it reports.

**Paraphrase equivalence.** `semantic_paraphrase` is label-preserving only if the rewrite really
preserves meaning. `experiments/equivalence_guard.py` enforces this with bidirectional NLI
entailment and raises `GuardUnavailable` rather than falling back to a weaker check. Cosine
similarity is not usable for this: measured on these corpora it ranks negations (0.825–0.924)
above a genuine paraphrase (0.706), which is the wrong order for an equivalence test.

---

## Repository structure

```
experiments/                      # evaluation and verification code
  attack_semantics.py             # label-preserving vs label-changing; picks VS or DFR
  equivalence_guard.py            # NLI entailment gate for paraphrase attacks
  attack_generators.py            # attack primitives (Sybil, LLM-generated, negation, injection)
  eval_llm_victim.py              # LLM victims via an OpenAI-compatible endpoint
  eval_attack_comparison.py       # recurrent victims (LSTM / Bi-LSTM / RNN)
  eval_structured.py              # encoder victims (BERT / RoBERTa)
  eval_vulnerability.py           # Sybil and content-perturbation track
  eval_whitebox.py                # FGSM / PGD white-box bound
  victim_api.py                   # register your own fact-checker as a victim
  gate_llm_victims.py             # learnability gate for a registered victim
  gate_recurrent.py               # learnability gate for recurrent victims
  check_c3_victims.py             # per-checkpoint lift and skew verdicts
  vulncheck.py                    # is a reported rate interpretable at all
  verify_experiments.py           # static checks over the runners
  check_split_leakage.py          # train/test split leakage check
  check_gen_determinism.py        # generator determinism check
  check_execution_variance.py     # same-condition variance across executions
  check_single_source_attacks.py  # one definition per attack family
  data_loaders.py                 # dataset loaders (HuggingFace + local CSV)
  dl_models.py                    # recurrent victim definitions
  fetch_datasets.py               # download the corpora (none are bundled)
  preflight_check.py              # environment sanity checks

scripts/                          # resume-safe experiment drivers (see "Running experiments")

outputs/                          # released per-condition result CSVs
data/MANIFEST.json                # corpus provenance; the corpora themselves are fetched
environment.yml                   # conda environment
```

---
## Setup

### 1. Conda environment

```bash
conda create -n vulnfc python=3.12 -y
conda activate vulnfc

# Install PyTorch (CPU / Apple Silicon MPS)
pip install torch

# All dependencies (CPU is sufficient for every reported result)
conda env update -n vulnfc -f environment.yml
```

### 2. Ollama — local LLM (C1, C5)

```bash
ollama serve          # keep this terminal open
ollama pull llama3.2
# verify: curl http://localhost:11434/v1/models
```

### 3. macOS OMP guard (required for C3)

```bash
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

> **`OMP_NUM_THREADS=1` is not advisory (confirmed 2026-09-07).** Fine-tuning BERT/RoBERTa on
> `covid19` **segfaults (signal 11) at the first training step** with `OMP_NUM_THREADS=3`,
> reproducibly, on Intel macOS — before and after the dynamic-padding change, so it is the OpenMP
> threading itself and not memory pressure. TabFact tolerated 3 and even 5 threads, so the crash is
> corpus-dependent and you may not hit it immediately. Set the guard for every training run.
>
> Losing the extra threads costs less than it sounds: on this workload 5 threads bought only 17%
> over 3 (47 vs 55 s/step) because it is memory-bandwidth bound, not compute bound.

### 4. Datasets — **run this before any experiment**

> **Important (2026-08-11).** The synthetic fallback in `data_loaders.py` is now **OFF by default**.
> A missing dataset raises `FileNotFoundError` instead of silently substituting placeholder text.
> The fallback is off because a silent substitution is undetectable downstream: 262 of 658 earlier
> runs evaluated about 20 template sentences instead of real data and reported normal-looking
> numbers. Do not re-enable it for anything you intend to report.

Fetch every available dataset to a local CSV, with a provenance manifest:

```bash
# What it does: downloads each corpus once, normalises it to data/<name>/<name>.csv
# (columns: claim,label — 1 = TRUE/REAL, 0 = FAKE), and records source, retrieval date,
# row count, label distribution and SHA-256 in data/MANIFEST.json.
# Why: pinning to local CSVs makes runs reproducible regardless of HuggingFace Hub changes —
# `datasets >= 4.0` dropped script-based loaders, which is what silently broke liar,
# health_fact and covid_lies at experiment time.
python experiments/fetch_datasets.py --all

# preview without downloading
python experiments/fetch_datasets.py --all --dry_run
```

#### 4b. TabFact (no result)

The structured TabFact track **carries no result in the paper**: its victims do not clear the
learnability gate. The reason is recorded here because it determines how the corpus must be built.

The distributed `tabfact.csv` has columns `statement,label` and no tables. TabFact is a
*table*-based verification task, so without the table a statement such as "the total attendance
for week 8 is 48667" has no evidence to check against, and both encoders sit at chance:
`bert-base-uncased_tabfact` scores 0.5100 against a 0.5275 majority-class baseline — worse than
always predicting the majority class — and `roberta-base_tabfact` puts 378 of 400 predictions in
one class. A near-constant classifier cannot be flipped, so it scores as maximally robust on
Vulnerability Score, and any architecture comparison built on these checkpoints measures
prediction skew rather than robustness. The split was also statement-level rather than
table-level, which leaks table content from train into test.

Both faults are what the learnability gate (`check_c3_victims.py`) and `vulncheck.py` exist to
catch. No rebuilt corpus is shipped and no TabFact number is reported.

## Checking that a vulnerability number is interpretable (`vulncheck`)

`experiments/vulncheck.py` decides whether a poisoning benchmark's own rates can be read as
evidence of robustness. It does **not** load a model, run an attack, or need a GPU: every check is
a function of quantities a benchmark already computes, so it reads the released result CSVs and
runs in about 1.2 s over this repository's 1,744 reported rates.

**Why it exists.** A rate over flipped predictions gives a victim whose output ignores its input
the same score (0.000) as a victim that resists every attack. Three further failures hide in the
same place: a rate at its ceiling cannot vary, a difference below one claim cannot be separated
from a single item, and a mean over seeds can conceal an order-of-magnitude spread.

**Checks.** `D` target denominator present · `G` majority-class gate (lift and skew, with margins,
including victims gated *before* any attack) · `R` resolvability: a difference no larger than
max(1/n₁,1/n₂) is one claim wide · `S` ceiling/floor saturation · `V` per-cell spread across seeds ·
`P` stability of a *released* between-victim difference across the seeds behind it.

Checks whose inputs were never released are reported `n/a` and counted separately — never silently
as passes. That category is the point when reading someone else's benchmark.

```bash
conda activate vulnfc12

# Audit this repository's own outputs (the default glob)
python experiments/vulncheck.py qc_csv=outputs/c3_sweep/victim_qc.csv

# Audit any other benchmark's released CSVs
python experiments/vulncheck.py results_glob='path/to/bench/**/*.csv'

# Show passing cells too, and raise the report cap
python experiments/vulncheck.py show_ok=true max_report_rows=200

# Write the default config to conf/vulncheck.yaml (Hydra; override any field on the CLI)
python experiments/vulncheck.py --write-config
```

**Gotchas.**
- It only reads CSVs carrying a rate column (`vs`, `asr`, `attack_success_rate`, `any_flip_rate`,
  `flip_rate`); timing and training logs are ignored by design.
- `n_targets` is **not** accepted as the denominator. VS is a rate over the correct-set |T⁺|;
  `n_targets` is |T|, a different population, and |T⁺| is not recoverable from it. Our
  first-generation result files (`c1/c3/c5/c7/c8/c9/c10_results.csv`) record the rate but not |T⁺|,
  so the tool reports them as failing `D` — correctly.
- The gate needs a trivial baseline, so pass `qc_csv=` (a `victim_qc.csv`-shaped file with
  `accuracy`, `majority_baseline`, `skew`). Without it `G` is reported `n/a`, not passed.
- A QC file usually has one row per training seed; the gate verdict is aggregated
  **conservatively** (worst lift, worst skew) and the seed count is printed, so a degenerate seed
  cannot be waved through by a luckier one.
- Cells are grouped by the fields present **in** the CSVs. A factor recorded only in a *filename*
  (our sweep puts the learning rate there) cannot be seen, so conditions differing only in it merge
  into one cell and its spread covers both. Put every experimental factor in a column.
- **Exit status: 0 clean, 1 if anything failed, 2 on a usage error.** The non-zero exit is the
  intended use — run it in CI before the numbers reach a paper.

The saved reference run is `outputs/vulncheck/full_run.txt`.

## Testing your own fact-checking system

The benchmark is a harness, not only a study. A new fact-checking system can be tested by registering it
as a victim; the attack taxonomy, budgets, learnability gate and metrics then apply unchanged.

**The contract is one method.**

```python
def predict(self, claim: str, shot_mode: str) -> int | None:
    """1 = TRUE, 0 = FAKE, None = no parseable verdict."""
```

`None` is counted as *unavailable* and reported beside every rate. It is never dropped, because a
system that cannot answer is not a system that proved robust.

**Out-of-tree — no edit to this repository.** Put your system anywhere on `PYTHONPATH`:

```python
# mypackage/victims.py
class MySystem:
    def __init__(self, spec: str = ""):      # spec is free-form: endpoint, checkpoint, config
        self.endpoint = spec or "http://localhost:8000"
    def predict(self, claim, shot_mode):
        ...
```

then run any victim-facing script with a `py:` spec:

```bash
# gate it first — this is the check the paper argues no benchmark should skip
python experiments/gate_llm_victims.py \
    --victim-model "py:mypackage.victims:MySystem" \
    --corpora kaggle covid19 pubhealth --n-claims 1000 --k-folds 5 --write

# then attack it, with the same taxonomy and budgets used in the paper
python experiments/eval_llm_victim.py \
    --victim_model "py:mypackage.victims:MySystem:https://my-endpoint" \
    --dataset covid19 --shot_mode zero_shot
```

**In-tree**, if you prefer a short name:

```python
from victim_api import register_victim

@register_victim("my_system")
class MySystem: ...
```

then `--victim_model my_system` or `--victim_model my_system:<spec>`.

**Read the gate verdict before the attack numbers.** A victim that fails it (lift over its
majority-class baseline $\le 0.05$, or one class holding $\ge 90\%$ of predictions) produces
flip-rates that cannot distinguish robustness from inability to do the task. `gate_llm_victims.py`
reports `PASS`, `FAIL` or `INDETERMINATE` with the five-fold lift interval, and prints the margin so
a different threshold can be applied to the released table.

**Gotchas.** A bad spec raises at startup rather than recording every claim as unavailable, so a
misconfigured run fails loudly. A model tag containing a colon (`llama3.2:latest`) is *not* treated
as a plugin spec; only registered names and the `py:` prefix are. Importing a module executes it, so
the import path must come from your own command line — never from a dataset, a claim, a model
response or any file the benchmark downloads.

## Running experiments

All scripts are **resume-safe**: they append to existing CSVs and skip completed conditions on
relaunch. Each takes `PY`, `API` and `GEN` from the environment, so nothing is hard-coded to one
machine:

```bash
conda activate vulnfc
export PY=python                               # interpreter to use
export API=http://127.0.0.1:11434/v1           # OpenAI-compatible endpoint
export GEN=llama3.2:latest                     # attack generator
```

These are the drivers included here, and the runner each one calls:

| Script | Runner | Grid |
|---|---|---|
| `run_exp_C1_v2.sh` | `eval_llm_victim.py` | llama3.2 × 4 corpora × 3 attacks × 3 seeds |
| `run_exp_C1_v3.sh` | `eval_llm_victim.py` | 3 LLM victims × 3 corpora × 3 seeds |
| `run_exp_C9_v2.sh` | `eval_attack_comparison.py` | recurrent victims, `llm_generated` + `pos_negation` |
| `run_rerun_llm_track.sh` | `eval_llm_victim.py` | re-measures the C1 v2 and v3 conditions under the corrected label semantics |
| `run_rerun_common_grid.sh` | `eval_structured.py`, `eval_attack_comparison.py` | encoders (2 × 2 corpora × 3 train × 3 attack seeds) + recurrent (3 × 2 × 3) |
| `run_C1_v3b_phi_split.sh` | `eval_llm_victim.py` | phi3.5 split re-run, one corpus per invocation |
| `probe_phi_kaggle_split.sh` | `eval_llm_victim.py` | single-condition probe |
| `requeue_degraded_C1_v3.sh` | — | re-queues conditions the data-quality guard dropped |
| `export_checkpoints.sh` | — | packages trained victims for transfer |

The two `run_rerun_*` drivers are the ones to use for a re-measurement; they take the population
flags from the published runs (`--max_samples 200` for C1 v2; `--max_samples 600
--n_nontarget_scored 60` for C1 v3) so a re-measured cell is comparable to the number it replaces.
Changing the population would change the denominator as well as the metric, and nothing could then
be attributed to the metric.

Encoder cells retrain their victim: checkpoints are ~154 GB and are not redistributable, so
`eval_structured.py` trains at the stated seed and hyperparameters rather than loading one. A
retrained victim is a fresh measurement, not a re-scoring, and each is put through the learnability
gate before its numbers are used.

### Quick smoke test (~10 s, no GPU/API)

```bash
python3 experiments/eval_vulnerability.py \
  --dataset kaggle --model bert --attack_type sybil \
  --budget 0.02 --seed 42 --max_samples 5 \
  --output_csv /tmp/smoke.csv --timing_csv /tmp/smoke_timing.csv
```

### The retrieve-then-verify track (not included)

A retrieval-then-verdict pipeline over a FAISS store was explored during development. It
produces **no number reported in the paper**: no table reads from it, and none of its result
files are released. Its evaluator, index builder, driver script and result CSVs are therefore
not part of this artifact, and nothing here depends on them. The track is recorded only so that
its absence is deliberate rather than an omission.

---

## Verification tools

Run these after any change to the paper or the results.

```bash
# Mechanical consistency audit of the LaTeX: metric arithmetic (F1 == 2PR/(P+R)),
# table->CSV provenance, references.bib hygiene, cross-reference integrity.
python experiments/audit_paper.py --tex paper-short.tex

# Regenerate the pre-attack table by INFERENCE ONLY from saved checkpoints.
# Emits binary AND macro P/R/F1 plus raw confusion matrices, so the averaging
# convention is explicit — the original table mixed two conventions in one row.
python experiments/recompute_baseline_metrics.py \
    --output_csv outputs/baseline_metrics/baseline_metrics.csv
# the released reference is outputs/baseline_metrics/baseline_metrics_real.csv

# Hold every victim to a learnability gate before attacking it: accuracy must beat the
# majority-class baseline by >0.05 and no class may hold >90% of predictions. A degenerate
# victim is unflippable and so scores as maximally robust on every rate-based measure.
# Exits 2 if any checkpoint fails. Run it on partial results — don't wait for a full sweep.
python experiments/check_c3_victims.py --csv outputs/c3_sweep/victim_qc.csv

# Fail if an attack definition exists in more than one module. Exits 1 on any duplicate.
python experiments/check_single_source_attacks.py
```

`audit_paper.py` is read-only and safe to run at any time. All are CPU-only.

### Applying the gate to the LLM victims (`gate_llm_victims.py`)

`check_c3_victims.py` gates the encoder checkpoints from a QC file. The LLM victims have no such
file: the released per-condition rows record no class balance for the subset each condition scored,
so accuracy there cannot be compared against any baseline recovered from the full split. That is an
unmatched comparison. `gate_llm_victims.py` closes it by measuring accuracy, majority baseline and
prediction skew over **one** population — the same test split the harness uses, scored through the
harness's own `APIVictim.predict`.

The victim is zero-shot, so there is nothing to train: the `k` folds put an interval on the lift,
each fold against its own baseline. A verdict that flips inside that interval prints
`INDETERMINATE` rather than a pass or a fail. Rows are appended per corpus as each finishes.

```bash
# Local. Keep --workers at or below the server's parallel slots (see the warning below).
python experiments/gate_llm_victims.py --n-claims 1000 --k-folds 5 --workers 1 --write
```

**Why a local run can stall.** `APIVictim` uses `timeout=180.0, max_retries=3`. Generation is
capped at 10 tokens, so a 3-minute response is never slow decoding — it is a request queued behind
others. A default Ollama serves `OLLAMA_NUM_PARALLEL=1`, so any other process using the same server
serializes against this one and each stall costs up to 4x180 s for a single claim. Check for
`500 | 3m0s` lines in `~/.ollama/logs/server.log` before trusting a slow run's throughput.

#### Full test splits on a CPU VM

The full splits are 12,537 claims and `kaggle` carries ~97% of the cost (8,119 claims averaging
~621 prompt tokens, against ~21-46 for the other three). The work is prompt-eval bound for a 3B Q4
model emitting 10 tokens, so it wants cores and memory bandwidth; **a GPU buys nothing here.** Use
an on-demand instance — a spot instance was preempted 26 minutes in.

```bash
gcloud compute instances create vulnfc-gate2 \
    --zone=us-central1-a --machine-type=n2-standard-32 \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=50GB --boot-disk-type=pd-balanced

# On the VM: Ollama, then four slots and a context above the longest claim.
curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_KEEP_ALIVE=24h"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama pull llama3.2:latest        # retry once: this races a fresh daemon at boot

python3 -m venv .venv
.venv/bin/pip install openai datasets numpy scipy pandas
.venv/bin/python -u experiments/gate_llm_victims.py \
    --n-claims 20000 --k-folds 5 --workers 4 --write
```

Three things decide whether the result is comparable at all, and each must be checked, not assumed:

- **Upload `data/{covid19,politifact,pubhealth}/*.csv`.** Only `kaggle` loads from HuggingFace.
  `covid_lies` no longer exists there and `liar`/`health_fact` are loader scripts, unsupported in
  `datasets>=4.0`, so those three fall back to local CSVs. Without them the VM scores a different
  population, or drops the corpus entirely.
- **Verify the split sizes** before scoring: 8119 / 2141 / 259 / 2018 for
  kaggle / covid19 / politifact / pubhealth at `seed=42`. A mismatch means a different population.
- **Verify the model digest matches**, not just the tag. `llama3.2:latest` is floating; `ollama list`
  must show the same ID on both machines (`a80c4f17acd5` for the reported runs).
- **Set `OLLAMA_CONTEXT_LENGTH` above the longest claim.** The longest kaggle article is ~12,208
  tokens; a default context truncates it silently and changes the prediction.

Confirm the victim answers one known claim before scoring thousands. A server with no model pulled
returns `None` for every request, which the script counts as `n_unavailable` rather than dropping —
a victim that cannot be queried is not a victim that passed.

### Which results file is authoritative

Most experiment directories hold several CSVs: `_preseed`, `_unseeded`, `_oldmetric`, `_partial` and
dated snapshots are kept rather than overwritten, so a superseded run can always be diffed against
the one that replaced it. That means **filename, not directory, decides what the paper reports.**

The table below is the authority. The generator that consumes these files lives with the
manuscript, not here: generated tables and figures, and the scripts that emit them, are excluded
from this deposit (see `.gitignore`), which ships the experiments and the result CSVs the tables
are computed from.

| Experiment | Authoritative file | Conditions |
|---|---|---|
| C1, LLM victim (attack comparison) | `c1_v2/c1_v2_results.csv` | 36 |
| C1 re-measured (label semantics) | `c1_v2/c1_v2_results_dfr.csv` | 24 |
| C2, recurrent victims | `c9_v2/c9_v2_results.csv` | 108 |
| C2 re-measured (label semantics) | `c9_v2/c9_v2_results_dfr.csv` | 18 |
| C3, encoder track | `c3_sweep/c7_fgsm_*`, `c8_sybil_*`, `c7_pgd_*` | 38 |
| C5, LLM victim scale sweep | `c1_v3/c1_v3_results.csv` | 80 (1 degraded row dropped) |
| C5 re-measured (label semantics) | `c1_v3/c1_v3_results_dfr.csv` | 54 |
| Crossed grid, encoders | `common_grid/llmgen_*.csv` | 36 |
| Crossed grid, pre-correction | `common_grid_published_vs/llmgen_*.csv` | 36 (archived) |
| C5 scope, source priors | `c6_v2/c6_v2_results_priors.csv` | 810 |
| C6, aggregation-layer defence | `c6_v2/c6_v2_results_5seed.csv` | 810 |
| | **total reported** | **1,882** |

The paper's condition count is produced from these files by the generator, never typed.

**Excluded, deliberately:** `c7_v2/` (12) and `c8_v2/` (27) are the retired TabFact encoder track;
`r1_1_sweep/` (24) is the R1-1 sweep. All three measured victims that failed the learnability
gate, so none of them carries a reported number.

### One definition per attack

`attack_generators.py` holds the only definition of `_INJECTION_SUFFIXES` — the five
prompt-injection templates the paper reports — and of the word substituted into `{target}`
(`FAKE`/`TRUE`). Import from it; never redefine either in a caller.

A private copy in a caller is a reportable defect even when it is never exercised: a user
passing `--attack_type prompt_injection` would get an attack the paper does not describe, which
would make the paper's description of its own instrument wrong. `check_single_source_attacks.py`
scans every module here and fails if a second definition appears.

No reported number rests on prompt injection against a structured victim: no `c8_*` file
contains it.

If you need a variant suffix set, add it to `attack_generators.py` under its own name and select
it explicitly.

---

## License

MIT — see `LICENSE`.
