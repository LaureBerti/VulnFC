#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
SEEDS=${SEEDS:-"42 123 456"}
NSEEDS=$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')
DATASETS=${C1_DATASETS:-"kaggle politifact covid19 pubhealth"}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/c1_v2 logs

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
echo "[C1] start $(date)"
for ds in $DATASETS; do for seed in $SEEDS; do
  for atk in llm_generated semantic_paraphrase prompt_injection; do
    if cond_done outputs/c1_v2/c1_v2_results.csv "$NSEEDS" "$seed" \
         "dataset=$ds" "attack_type=$atk"; then
      echo "  [skip] $ds/$atk/s$seed already done"; continue
    fi
    if nice -n 19 "$PY" experiments/eval_llm_victim.py \
         --dataset "$ds" --victim_model llama3.2:latest --attack_type "$atk" \
         --shot_mode zero_shot --seed "$seed" --budget 0.10 --max_samples 200 \
         --api_base http://localhost:11434/v1 --max_workers 4 \
         --output_csv outputs/c1_v2/c1_v2_results.csv \
         --timing_csv outputs/c1_v2/c1_v2_timing.csv \
         > logs/c1_last_condition.log 2>&1; then
      RAN=$((RAN+1)); grep -iE "VS=" logs/c1_last_condition.log | tail -1
    else
      echo "  [FAIL] $ds/$atk/s$seed:"; tail -3 logs/c1_last_condition.log | sed 's/^/        /'
      FAIL=$((FAIL+1))
    fi
  done
done; done
echo "[C1] done $(date) — ran $RAN, failed $FAIL"
