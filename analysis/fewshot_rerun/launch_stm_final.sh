#!/bin/bash
# Runs stm_final_units.txt after the first Rodney queue (launch.sh, its xargs pid given) has finished, same settings
# (4 at a time, 2 threads each). run_v2.sh skips units that already have job.done.
# Usage (repo root): nohup bash analysis/fewshot_rerun/launch_stm_final.sh <xargs pid> > runs_local/fewshot_rodney_20261014/launch_stm_final.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PY=../.venv/bin/python
D=runs_local/fewshot_rodney_20261014
echo "[$(date '+%F %T')] waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] start, code $(git rev-parse --short HEAD)"
tr '\n' '\0' < $D/stm_final_units.txt | xargs -0 -P 4 -I{} bash -c '
  e=${1%%|*}; a=${1#*|}
  if eval "env $e bash run_v2.sh $a" > /dev/null 2>&1; then r=done; else r=FAILED; fi
  echo "[$(date "+%F %T")] $r $1"' _ {}
echo "[$(date '+%F %T')] finished"
