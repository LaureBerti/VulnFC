#!/usr/bin/env bash
set -uo pipefail
cd "${VULNFC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$PWD/experiments"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY="${PYTHON:-python3}"
API=http://127.0.0.1:11434/v1
GEN=llama3.2:latest          # the generator the published runs used; stated in the paper
WORKERS=4                    # matches OLLAMA_NUM_PARALLEL, verified at setup

mkdir -p out archive logs
C1V2=out/c1_v2_results.csv
C1V3=out/c1_v3_results.csv

done_already() {   # csv, dataset, victim, attack, seed
  [ -f "$1" ] || return 1
  awk -F, -v d="$2" -v v="$3" -v a="$4" -v s="$5" \
    'NR>1 && $1==d && $2==v && $3==a && $5==s {found=1} END{exit !found}' "$1"
}

run_one() {        # csv, track, dataset, victim, attack, seed, extra-flags...
  local csv=$1 track=$2 ds=$3 vic=$4 atk=$5 sd=$6; shift 6
  local pop=("$@")
  local tag="${track}_${ds}_${vic//[:.]/-}_${atk}_s${sd}"
  if done_already "$csv" "$ds" "$vic" "$atk" "$sd"; then
    echo "  SKIP $tag (already in $csv)"; return 0
  fi
  echo "  RUN  $tag"
  $PY experiments/eval_llm_victim.py \
      --dataset "$ds" --victim_model "$vic" --attack_type "$atk" \
      --shot_mode zero_shot --seed "$sd" --max_workers "$WORKERS" \
      --api_base "$API" --attack_model "$GEN" \
      "${pop[@]}" \
      --archive_attacks "archive/${tag}.jsonl" \
      --output_csv "$csv" --timing_csv "${csv%.csv}_timing.csv" \
      > "logs/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAIL $tag rc=$rc  (see logs/${tag}.log)" | tee -a logs/FAILURES.txt
  else
    grep -oE '(VS|DFR)=[0-9.]+( \(trivial [0-9.]+\))?' "logs/${tag}.log" | tail -1 | sed 's/^/        /'
  fi
}

echo "=== c1_v2: llama3.2, four corpora ==="
for ds in kaggle covid19 politifact pubhealth; do
  for atk in llm_generated semantic_paraphrase; do
    for sd in 42 123 456; do
      run_one "$C1V2" c1v2 "$ds" llama3.2:latest "$atk" "$sd" \
              --budget 0.10 --max_samples 200
    done
  done
done

echo "=== c1_v3: three victims, three corpora ==="
for vic in llama3.2:latest phi3.5:latest qwen2.5:14b; do
  for ds in kaggle covid19 pubhealth; do
    for atk in llm_generated semantic_paraphrase; do
      for sd in 42 123 456; do
        run_one "$C1V3" c1v3 "$ds" "$vic" "$atk" "$sd" \
                --budget 0.10 --max_samples 600 --n_nontarget_scored 60
      done
    done
  done
done

echo "CAMPAIGN_DONE" > CAMPAIGN_STATUS
echo "=== complete: $(wc -l < $C1V2 2>/dev/null) + $(wc -l < $C1V3 2>/dev/null) rows (incl. headers) ==="
