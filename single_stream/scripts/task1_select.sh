#!/bin/bash
# Task-1 round 1 selection queue for one LTM variant at c = 1 (plan.md section 6, step 2).
# Usage: bash single_stream/scripts/task1_select.sh <out_root> <tag> "<net args>" <lr> [<lr> ...]
set -u
OUT=$1; TAG=$2; NET=$3; shift 3
PY=/home/incubator/dev/claude/.venv/bin/python
cd "$(dirname "$0")/../.."
for LR in "$@"; do
  D=$OUT/${TAG}_c1_lr${LR}_s0
  echo "$(date '+%F %T') start $D"
  $PY -m single_stream.run_stream $NET --cluster-size 1 --epochs 5 --lr $LR --val-holdout 600 \
      --seed 0 --threads 2 --out $D > /dev/null 2>&1 || echo "$(date '+%F %T') FAILED $D"
  echo "$(date '+%F %T') done $D: $(tail -1 $D/log.txt)"
done
