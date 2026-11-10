#!/usr/bin/env bash
set -uo pipefail
cd "${VULNFC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$PWD/experiments"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
PY="${PYTHON:-python3}"
API=http://127.0.0.1:11434/v1
GEN=llama3.2:latest

mkdir -p out/common_grid out/c9_v2 logs archive

enc_done() { [ -s "out/common_grid/llmgen_$1.csv" ]; }

echo "=== encoders: eval_structured, 36 cells ==="
for M in bert roberta; do
  for DS in covid19 pubhealth; do
    for TSEED in 42 123 456; do
      CK="checkpoints/${M}_${DS}_s${TSEED}"
      for ASEED in 42 123 456; do
        NAME="${M}_${DS}_t${TSEED}_a${ASEED}"
        if enc_done "$NAME"; then echo "  SKIP $NAME"; continue; fi
        echo "  RUN  $NAME"
        $PY experiments/eval_structured.py \
            --dataset "$DS" --model "$M" --attack_type llm_generated \
            --checkpoint_dir "$CK" --budget 0.10 --seed "$ASEED" --max_samples 400 \
            --api_base "$API" --attack_model "$GEN" \
            --output_csv "out/common_grid/llmgen_${NAME}.csv" \
            --timing_csv "out/common_grid/timing_${NAME}.csv" \
            > "logs/grid_${NAME}.log" 2>&1 \
          && grep -oE '(VS|DFR)=[0-9.]+( \(trivial [0-9.]+\))?' "logs/grid_${NAME}.log" | tail -1 | sed 's/^/        /' \
          || { echo "  FAIL $NAME" | tee -a logs/GRID_FAILURES.txt; tail -3 "logs/grid_${NAME}.log" | sed 's/^/        /'; }
      done
    done
  done
done

echo "=== recurrent: eval_attack_comparison, 18 conditions ==="
C9=out/c9_v2/c9_v2_results.csv
for M in lstm_title bi_lstm rnn; do
  for DS in covid19 pubhealth; do
    for SEED in 42 123 456; do
      if [ -f "$C9" ] && awk -F, -v d="$DS" -v m="$M" -v s="$SEED" \
           'NR>1 && $1==d && $2==m && $3=="llm_generated" && $4==s {f=1} END{exit !f}' "$C9"; then
        echo "  SKIP ${M}_${DS}_s${SEED}"; continue
      fi
      echo "  RUN  ${M}_${DS}_s${SEED}"
      $PY experiments/eval_attack_comparison.py \
          --dataset "$DS" --model "$M" --attack_type llm_generated \
          --budget 0.10 --seed "$SEED" --max_samples 400 \
          --api_base "$API" --attack_model "$GEN" \
          --output_csv "$C9" --timing_csv "out/c9_v2/timing.csv" \
          > "logs/c9_${M}_${DS}_s${SEED}.log" 2>&1 \
        && grep -oE '(VS|DFR)=[0-9.]+( \(trivial [0-9.]+\))?' "logs/c9_${M}_${DS}_s${SEED}.log" | tail -1 | sed 's/^/        /' \
        || { echo "  FAIL ${M}_${DS}_s${SEED}" | tee -a logs/GRID_FAILURES.txt; tail -3 "logs/c9_${M}_${DS}_s${SEED}.log" | sed 's/^/        /'; }
    done
  done
done

echo "GRID_DONE" > GRID_STATUS
echo "=== complete: $(ls out/common_grid/llmgen_*.csv 2>/dev/null | wc -l) encoder cells, $(( $(wc -l < $C9 2>/dev/null || echo 1) - 1 )) recurrent rows ==="
