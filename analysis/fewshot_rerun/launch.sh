#!/bin/bash
# Runs units.txt (environment|run_v2.sh arguments), 4 at a time, 2 threads each, in file order (the supervised STM
# head's selection first). run_v2.sh skips units that already have job.done.
# Usage (repo root): nohup bash analysis/fewshot_rerun/launch.sh > runs_local/fewshot_rodney_20261014/launch.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PY=../.venv/bin/python
D=runs_local/fewshot_rodney_20261014
echo "[$(date '+%F %T')] start, code $(git rev-parse --short HEAD)"
tr '\n' '\0' < $D/units.txt | xargs -0 -P 4 -I{} bash -c '
  e=${1%%|*}; a=${1#*|}
  if eval "env $e bash run_v2.sh $a" > /dev/null 2>&1; then r=done; else r=FAILED; fi
  echo "[$(date "+%F %T")] $r $1"' _ {}
echo "[$(date '+%F %T')] finished"
