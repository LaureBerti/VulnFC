#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
SEEDS=${SEEDS:-"42 123 456"}
NSEEDS=$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')
DATASETS=${C9_DATASETS:-"kaggle politifact covid19 pubhealth climatefever tiger"}
MODELS=${C9_MODELS:-"lstm_title bi_lstm rnn"}
WORKERS=${WORKERS:-4}
OUT=outputs/c9_v2/c9_v2_results.csv
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/c9_v2 logs

cond_done() {
  local csv="$1" need="$2" seed="$3"; shift 3
  [ -f "$csv" ] || return 1
  python3 - "$csv" "$need" "$seed" "$@" <<'PYEOF'
import csv, sys
path, need, seed, pairs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
want = dict(p.split("=", 1) for p in pairs)
n = 0
try:
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if not all(str(row.get(k, "")).strip() == v for k, v in want.items()):
                continue
            n += 1
            if str(row.get("seed", "")).strip() == seed:
                sys.exit(0)
except Exception:
    sys.exit(1)
sys.exit(0 if n >= need else 1)
PYEOF
}

FAIL=0; RAN=0
echo "[C9] start $(date) — workers=$WORKERS"
for ds in $DATASETS; do for seed in $SEEDS; do for m in $MODELS; do
  for atk in llm_generated pos_negation; do
    if cond_done "$OUT" "$NSEEDS" "$seed" "dataset=$ds" "model=$m" "attack_type=$atk"; then
      echo "  [skip] $ds/$m/$atk/s$seed"; continue
    fi
    if nice -n 19 "$PY" experiments/eval_attack_comparison.py \
         --dataset "$ds" --model "$m" --attack_type "$atk" \
         --budget 0.10 --seed "$seed" --max_samples 400 \
         --api_base http://localhost:11434/v1 --attack_model llama3.2:latest \
         --max_workers "$WORKERS" \
         --output_csv "$OUT" \
         --timing_csv outputs/c9_v2/c9_v2_timing.csv \
         > logs/c9_last_condition.log 2>&1; then
      RAN=$((RAN+1)); grep -iE "VS=" logs/c9_last_condition.log | tail -1
    else
      echo "  [FAIL] $ds/$m/$atk/s$seed:"; tail -3 logs/c9_last_condition.log | sed 's/^/        /'
      FAIL=$((FAIL+1))
    fi
  done
done; done; done
echo "[C9] done $(date) — ran $RAN, failed $FAIL"
