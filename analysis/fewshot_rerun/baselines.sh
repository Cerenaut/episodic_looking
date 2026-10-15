#!/bin/bash
# Starting points for the few-shot panels (Dave, 15 Oct 2026): the pre-continual evaluation (four test/val sets) of
# each RL STM final seed's pre-trained checkpoint (runs_v2_fs_rl/rl_lr0.1/baseline/rl/pair0_1/seed<k>; the finals ran
# pre-training only) and of the LTM itself (runs_v2/baseline/ltm/pair0_1). Heads write theirs in every unit.
# Rodney CPU, 4 at a time, 2 threads each. Usage (repo root): nohup bash analysis/fewshot_rerun/baselines.sh > runs_local/fewshot_final_20261015/baselines.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PY=../.venv/bin/python
echo "[$(date '+%F %T')] start, code $(git rev-parse --short HEAD)"
{ for s in 1 2 3 4 5 6 7 8 9 10; do echo "RUNS=runs_v2_fs_rl/rl_lr0.1 LR=0.1|baseline rl \"0 1\" $s"; done
  echo "RUNS=runs_v2|baseline ltm \"0 1\" 0"; } | tr '\n' '\0' | xargs -0 -P 4 -I{} bash -c '
  e=${1%%|*}; a=${1#*|}
  if eval "env $e bash run_v2.sh $a" > /dev/null 2>&1; then r=done; else r=FAILED; fi
  echo "[$(date "+%F %T")] $r $1"' _ {}
echo "[$(date '+%F %T')] finished"
