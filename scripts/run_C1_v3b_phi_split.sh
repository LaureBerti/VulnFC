#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
LIVE=experiments/eval_llm_victim.py
STAGED=staged/eval_llm_victim_split.py
OUT=outputs/c1_v3b/c1_v3b_results.csv
TIMING=outputs/c1_v3b/c1_v3b_timing.csv
STAMP=$(date +%Y%m%d_%H%M%S)
VICTIM=phi3.5:latest
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/c1_v3b logs

echo "[v3b] $(date) waiting for the C1 v3 pass and the requeue job to finish"
while pgrep -f "run_exp_C1_v3.sh" >/dev/null \
   || pgrep -f "requeue_degraded_C1_v3.sh" >/dev/null \
   || pgrep -f "eval_llm_victim.py" >/dev/null; do
  sleep 300
done
echo "[v3b] $(date) machine free"

cp -p "$LIVE" "experiments/eval_llm_victim_prev_${STAMP}.py"
cp -p "$STAGED" "$LIVE"
echo "[v3b] installed split-counter evaluator (previous kept as eval_llm_victim_prev_${STAMP}.py)"

run_one() {  # dataset attack seed
  nice -n 19 "$PY" "$LIVE" \
    --dataset "$1" --victim_model "$VICTIM" --attack_type "$2" \
    --shot_mode zero_shot --seed "$3" --budget 0.10 \
    --max_samples 600 --n_nontarget_scored 60 --max_workers 4 \
    --api_base http://localhost:11434/v1 --attack_model llama3.2:latest \
    --output_csv "$OUT" --timing_csv "$TIMING" \
    > logs/c1v3b_last_condition.log 2>&1
}

echo "[v3b] $(date) probe: $VICTIM/pubhealth/prompt_injection/s42"
if ! run_one pubhealth prompt_injection 42; then
  echo "[v3b] PROBE FAILED — evaluator error, not re-running the rest:"; tail -20 logs/c1v3b_last_condition.log
  exit 1
fi
grep -E "WARNING|unavailable" logs/c1v3b_last_condition.log || true

"$PY" - "$OUT" <<'PYEOF' || { echo "[v3b] probe contradicts the diagnosis — STOPPING for review"; exit 2; }
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))[-1]
clean, adv = int(r["n_unavail_clean"]), int(r["n_unavail_adv"])
tot = clean + adv
print(f"[v3b] probe split: {clean} clean-pass, {adv} adversarial-pass, refusal_rate={r['refusal_rate']}")
if tot == 0:
    print("[v3b] probe lost no items at all — inconclusive, review before continuing"); sys.exit(2)
sys.exit(0 if adv / tot >= 0.25 else 2)
PYEOF

FAIL=0; RAN=1
for seed in 42 123 456; do
  for ds in kaggle pubhealth covid19; do
    for atk in llm_generated semantic_paraphrase prompt_injection; do
      [ "$seed$ds$atk" = "42pubhealthprompt_injection" ] && continue   # the probe
      t0=$(date +%s)
      if run_one "$ds" "$atk" "$seed"; then
        RAN=$((RAN+1))
        echo "  [ok]  $ds/$atk/s$seed  $(( $(date +%s) - t0 ))s  $(grep -oE 'VS=[0-9.]+' logs/c1v3b_last_condition.log | tail -1)  $(grep -oE '\([0-9]+ clean-pass, [0-9]+ adversarial-pass\)' logs/c1v3b_last_condition.log | tail -1)"
      else
        FAIL=$((FAIL+1)); echo "  [FAIL] $ds/$atk/s$seed:"; tail -3 logs/c1v3b_last_condition.log | sed 's/^/         /'
      fi
    done
  done
done
echo "[v3b] done $(date) — ran $RAN, failed $FAIL"
