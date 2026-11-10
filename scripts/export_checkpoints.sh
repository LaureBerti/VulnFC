#!/usr/bin/env bash
set -euo pipefail
SRC=${1:?usage: export_checkpoints.sh <src checkpoints dir> <dest dir>}
DST=${2:?usage: export_checkpoints.sh <src checkpoints dir> <dest dir>}
MODELS=(bert-base-uncased_tabfact roberta-base_tabfact)

mkdir -p "$DST"
for m in "${MODELS[@]}"; do
  if [ ! -d "$SRC/$m" ]; then
    echo "missing: $SRC/$m — cannot export" >&2; exit 1
  fi
  mkdir -p "$DST/$m"
  find "$SRC/$m" -maxdepth 1 -type f -exec cp -p {} "$DST/$m/" \;
  echo "exported $m ($(du -sh "$DST/$m" | cut -f1))"
done

cat > "$DST/SHA256SUMS.txt" <<EOF
EOF
( cd "$DST" && find . -type f ! -name SHA256SUMS.txt -exec shasum -a 256 {} \; >> SHA256SUMS.txt )
echo "wrote $DST/SHA256SUMS.txt"
echo "total: $(du -sh "$DST" | cut -f1)"
