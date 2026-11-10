#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/vulnfc12/bin/python}
CSV=outputs/c1_v3/c1_v3_results.csv
QUAR=outputs/c1_v3/c1_v3_degraded.csv
STAMP=$(date +%Y%m%d_%H%M%S)

DS=kaggle; VICTIM=phi3.5:latest; ATK=prompt_injection; SEED=123

echo "[requeue] $(date) waiting for any active C1 v3 pass to finish"
while pgrep -f "run_exp_C1_v3.sh" > /dev/null || pgrep -f "eval_llm_victim.py.*c1_v3" > /dev/null; do
  sleep 300
done
echo "[requeue] $(date) machine free"

cp -p "$CSV" "${CSV%.csv}_prerequeue_${STAMP}.csv"
echo "[requeue] backed up to ${CSV%.csv}_prerequeue_${STAMP}.csv"

"$PY" - "$CSV" "$QUAR" "$DS" "$VICTIM" "$ATK" "$SEED" <<'PYEOF'
import csv, os, sys
csv_path, quar_path, ds, victim, atk, seed = sys.argv[1:7]
with open(csv_path) as fh:
    rd = csv.DictReader(fh); fields = rd.fieldnames; rows = list(rd)

def is_target(r):
    return (r["dataset"] == ds and r["llm_victim"] == victim
            and r["attack_type"] == atk and str(r["seed"]) == seed)

hit = [r for r in rows if is_target(r)]
keep = [r for r in rows if not is_target(r)]
if not hit:
    print("[requeue] no matching row — already requeued; nothing to move"); sys.exit(0)

new = not os.path.exists(quar_path)
with open(quar_path, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    w.writerows(hit)
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(keep)
print(f"[requeue] moved {len(hit)} row(s) to {quar_path}; {len(keep)} remain")
PYEOF

echo "[requeue] $(date) re-running $VICTIM/$DS/$ATK/s$SEED"
SEEDS="$SEED" VICTIMS="$VICTIM" CORPORA="$DS" ATTACKS="$ATK" \
  scripts/run_exp_C1_v3.sh
echo "[requeue] $(date) done"
