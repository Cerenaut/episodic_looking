#!/bin/bash
# Task-1 cluster-size sweep for one LTM variant at its selected step size and epochs (plan.md section 6, step 2).
# Usage: bash single_stream/scripts/task1_csweep.sh <out_root> <tag> "<net args>" <epochs> "<c:lr> ..."
#   e.g. "16:0.003 64:0.003 128:0.003 128:0.001 128:0.0003"
set -u
OUT=$1; TAG=$2; NET=$3; EPOCHS=$4; RUNS=$5
PY=/home/incubator/dev/claude/.venv/bin/python
cd "$(dirname "$0")/../.."
for CL in $RUNS; do
  C=${CL%%:*}; LR=${CL##*:}
  D=$OUT/${TAG}_c${C}_lr${LR}_s0
  echo "$(date '+%F %T') start $D"
  $PY -m single_stream.run_stream $NET --cluster-size $C --epochs $EPOCHS --lr $LR --val-holdout 600 \
      --seed 0 --threads 2 --out $D > /dev/null 2>&1 || echo "$(date '+%F %T') FAILED $D"
  echo "$(date '+%F %T') done $D: $(tail -1 $D/log.txt)"
done
