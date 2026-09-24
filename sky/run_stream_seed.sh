#!/bin/bash
# One seed of the seeded single-stream round: fine classes 3, 4 and 5 in parallel at the
# paper's full 96,000-exposure budget, batch size 1, lr 0.01.
#
# Why: the matched-budget head comparison (2026-09-24) rests on ONE unseeded single-stream
# run per class against three seeds per head, so the remaining statistical weakness is now on
# our side of that comparison. This closes it.
#
# Evaluated at the POLICY MEAN, not sampled. The heads are deterministic, so the mean is the
# footing the head comparison is made on; the sampled reading needed for figure 10 stays with
# the original unseeded run, and can be recovered from the checkpoints this saves anyway.
# Usage: ./sky/run_stream_seed.sh <seed>
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEED=${1:?seed}
CK=../cifar_100_pretrain
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt12.pth
ROOT=runs_stream_seed/s$SEED
LOG=runs_local/stream_seed
mkdir -p "$LOG" "$ROOT" "$CK/variants"
step() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG/progress.log"; }

step "single-stream seed $SEED: classes 3,4,5 in parallel, 96k exposures, lr 0.01, mean bias"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | tee -a "$LOG/progress.log"

rc=0
for c in 3 4 5; do
  [ -f "$LOG/s${SEED}_c${c}.done" ] && continue
  ( $PY cifar_main_stm_training.py --experiment-type few-shot --fine-classes $c --coarse-classes 0 1 \
        --batch-size 1 --training-steps 8000 --epochs 96 --evaluate-epochs 8 --evaluate-steps 800 \
        --eval-bias mean --learning-rate 0.01 --seed "$SEED" \
        --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root "$ROOT" \
        --stm-checkpoint-out $CK/variants/stm_stream_s${SEED}_c${c}.pth \
        > "$LOG/s${SEED}_c${c}.log" 2>&1 \
      && touch "$LOG/s${SEED}_c${c}.done" && step "seed $SEED class $c finished" \
      || { step "FAILED: seed $SEED class $c"; exit 1; } ) &
done
wait || rc=1
[ $rc -ne 0 ] && { step "JOB FAILED: seed $SEED"; exit 1; }
touch "$LOG/JOB_COMPLETE"
step "JOB DONE: single-stream seed $SEED"
