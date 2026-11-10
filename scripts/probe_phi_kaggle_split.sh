#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
OUT=outputs/c1_v3b/c1_v3b_results.csv
TIMING=outputs/c1_v3b/c1_v3b_timing.csv
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/c1_v3b logs

wait_ready() {
  local tries=0
  while pgrep -f "eval_llm_victim.py" >/dev/null; do sleep 120; done
  while [ $tries -lt 30 ]; do
    if "$PY" - <<'PYEOF'
import sys
from openai import OpenAI
c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=120.0, max_retries=1)
try:
    r = c.chat.completions.create(model="phi3.5:latest",
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=8, temperature=0.0)
    txt = (r.choices[0].message.content or "").strip()
    print(f"[ready] victim replied {txt!r}")
    sys.exit(0 if txt else 1)
except Exception as e:
    print(f"[ready] not ready: {type(e).__name__}: {e}"); sys.exit(1)
PYEOF
    then return 0; fi
    tries=$((tries+1)); echo "[ready] retry $tries/30 in 60s"; sleep 60
  done
  echo "[ready] victim never became ready — aborting rather than probing a dead service"; return 1
}

echo "[probe] $(date) checking victim readiness"
wait_ready || exit 1

for seed in 123 456; do
  echo "[probe] $(date) phi3.5/kaggle/prompt_injection/s$seed"
  t0=$(date +%s)
  if nice -n 19 "$PY" experiments/eval_llm_victim.py \
       --dataset kaggle --victim_model phi3.5:latest --attack_type prompt_injection \
       --shot_mode zero_shot --seed "$seed" --budget 0.10 \
       --max_samples 600 --n_nontarget_scored 60 --max_workers 4 \
       --api_base http://localhost:11434/v1 --attack_model llama3.2:latest \
       --output_csv "$OUT" --timing_csv "$TIMING" \
       > "logs/probe_s${seed}.log" 2>&1; then
    echo "  [ok] $(( $(date +%s) - t0 ))s  $(grep -oE 'WARNING: .*adversarial-pass\)' "logs/probe_s${seed}.log" | tail -1)"
  else
    echo "  [FAIL] s$seed:"; tail -5 "logs/probe_s${seed}.log" | sed 's/^/       /'
  fi
done

echo "[probe] $(date) summary"
"$PY" - "$OUT" <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["dataset"] == "kaggle" and r["attack_type"] == "prompt_injection"]
for r in rows:
    c, a = int(r["n_unavail_clean"]), int(r["n_unavail_adv"])
    t = c + a
    print(f"  seed {r['seed']:4} lost {t:3}  clean-pass {c:3} ({c/t:5.1%})  adversarial-pass {a:3} ({a/t:5.1%})"
          f"  refusal_rate={r['refusal_rate']}  VS={r['vs']}  n_tc={r['n_target_correct']}" if t else
          f"  seed {r['seed']:4} lost nothing")
PYEOF
