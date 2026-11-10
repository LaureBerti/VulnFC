#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
SEEDS=${SEEDS:-"42 123 456"}
VICTIMS=${VICTIMS:-"llama3.2:latest phi3.5:latest qwen2.5:14b"}
CORPORA=${CORPORA:-"kaggle pubhealth covid19"}
ATTACKS=${ATTACKS:-"llm_generated semantic_paraphrase prompt_injection"}
MAXS=${MAXS:-600}; NNON=${NNON:-60}; WORKERS=${WORKERS:-4}
OUT=outputs/c1_v3/c1_v3_results.csv
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/c1_v3 logs

cond_done() {
  local csv="$1"; shift
  [ -f "$csv" ] || return 1
  python3 - "$csv" "$@" <<'PYEOF'
import csv, sys
path, pairs = sys.argv[1], sys.argv[2:]
want = dict(p.split("=", 1) for p in pairs)
try:
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if all(str(row.get(k, "")).strip() == v for k, v in want.items()):
                sys.exit(0)
except Exception:
    pass
sys.exit(1)
PYEOF
}

FAIL=0; RAN=0; SKIP=0
echo "[C1v3] start $(date) — ${MAXS} samples, ${NNON} non-targets scored, ${WORKERS} workers"
for seed in $SEEDS; do
  echo "[C1v3] ===== seed $seed ====="
  for victim in $VICTIMS; do
    for ds in $CORPORA; do
      for atk in $ATTACKS; do
        if cond_done "$OUT" "dataset=$ds" "llm_victim=$victim" "attack_type=$atk" "seed=$seed"; then
          SKIP=$((SKIP+1)); echo "  [skip] $victim/$ds/$atk/s$seed"; continue
        fi
        t0=$(date +%s)
        if nice -n 19 "$PY" experiments/eval_llm_victim.py \
             --dataset "$ds" --victim_model "$victim" --attack_type "$atk" \
             --shot_mode zero_shot --seed "$seed" --budget 0.10 \
             --max_samples "$MAXS" --n_nontarget_scored "$NNON" --max_workers "$WORKERS" \
             --api_base http://localhost:11434/v1 --attack_model llama3.2:latest \
             --output_csv "$OUT" \
             --timing_csv outputs/c1_v3/c1_v3_timing.csv \
             > logs/c1v3_last_condition.log 2>&1; then
          RAN=$((RAN+1))
          echo "  [ok]  $victim/$ds/$atk/s$seed  $(( $(date +%s) - t0 ))s  $(grep -oE 'VS=[0-9.]+' logs/c1v3_last_condition.log | tail -1)"
        else
          FAIL=$((FAIL+1)); echo "  [FAIL] $victim/$ds/$atk/s$seed:"
          tail -3 logs/c1v3_last_condition.log | sed 's/^/         /'
        fi
      done
    done
  done
  echo "[C1v3] seed $seed complete at $(date) — ran $RAN, skipped $SKIP, failed $FAIL"
done
echo "[C1v3] done $(date) — ran $RAN, skipped $SKIP, failed $FAIL"
