#!/bin/bash
# RL single-stream learning-rate sweep (plan.md section 3, decision 5, "New 2 Oct"; results_v3.tex Setup): the one
# CLS/STM cell that is selected like the heads. Learning rates 0.00625, 0.01, 0.02; fine classes 3, 4, 5; pair 0 1;
# seed 1; validation split; capped at twice the draft's budget (384 epochs of 4,000 minibatch-1 steps). The budget
# is then chosen on the four-set validation curve with the heads' rule.
#
# 1. Pre-trains the RL STM once (12 epochs, batch 16, split) in the lr0.01 tree and copies that unit into the
#    other trees, so every learning rate starts from the same checkpoint.
# 2. Runs the 9 stream units NPROC at a time, fine class 3 first (all three learning rates), then 4, then 5.
# Resumable: run_v2.sh skips finished units (job.done) and refuses unfinished ones (archive them first).
# Usage: nohup caffeinate -i bash sweep_rl_stream_v2.sh > runs_v2_pilot_stm/sweep.out 2>&1 &
set -u
cd "$(dirname "$0")"
export PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
ROOT=${ROOT:-runs_v2_pilot_stm}
NPROC=${NPROC:-3}
LRS=${LRS:-"0.00625 0.01 0.02"}
CLASSES=${CLASSES:-"3 4 5"}
CAP=${CAP:-384}
mkdir -p "$ROOT"
LOG=$ROOT/sweep.log
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

step "sweep starting: lrs $LRS, classes $CLASSES, cap $CAP epochs, NPROC $NPROC, code $(git rev-parse --short HEAD)"
PRE=pretrain/rl/pair0_1/seed1
RUNS=$ROOT/rl_lr0.01 bash run_v2.sh pretrain rl "0 1" 1 || { step "pre-training FAILED"; exit 1; }
for lr in $LRS; do
  dest=$ROOT/rl_lr$lr/$PRE
  if [ ! -e "$dest" ]; then
    mkdir -p "$(dirname "$dest")" && cp -R "$ROOT/rl_lr0.01/$PRE" "$dest" && step "pre-training copied into $dest"
  fi
  [ -f "$dest/job.done" ] && [ -f "$dest/stm_pretrain.pth" ] || { step "pre-training missing in $dest"; exit 1; }
done

run_unit() {  # lr class
  if RUNS=$ROOT/rl_lr$1 LR=$1 EPOCHS=$CAP SAVE_STM=1 bash run_v2.sh stream rl "0 1" 1 "$2" < /dev/null; then
    step "done   lr $1 fine $2"
  else
    step "FAILED lr $1 fine $2"
  fi
}
for c in $CLASSES; do
  for lr in $LRS; do
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 30; done
    step "start  lr $lr fine $c"
    run_unit "$lr" "$c" &
    sleep 5
  done
done
wait

missing=0
for c in $CLASSES; do for lr in $LRS; do
  [ -f "$ROOT/rl_lr$lr/stream/rl/pair0_1/seed1/fine$c/job.done" ] || { missing=$((missing + 1)); step "missing: lr $lr fine $c"; }
done; done
if [ "$missing" = 0 ]; then step "SWEEP DONE"; else step "SWEEP FAILED: $missing units unfinished"; fi
