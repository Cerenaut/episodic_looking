#!/bin/bash
# Single-stream reproduction at the paper's full budget, with Dave's confirmed configuration (2026-09-23):
#   lr 0.01 (he scaled the minibatch-16 rate of 0.1 by the batch size, then simplified to 0.01), batch size 1,
#   STM pre-trained at minibatch 16 (his choice too), sampled evaluation as in the released code (NOT --eval-bias
#   mean, so these numbers are directly comparable with figure 10 of the paper; our other numbers use the mean and
#   read about 0.06 higher), 96,000 exposures = 96 epochs x 8000 batch-1 steps / 8 steps per episode.
#   Evaluation every 8 epochs = every 8,000 exposures, 12 points, matching the paper's figure cadence.
# This is the test of whether our earlier 12,000-exposure runs were simply an early snapshot of a slow curve.
# Results runs_stream_full/; logs runs_local/stream_full/. About 28 h for the three classes in parallel.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/stream_full
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
for c in 3 4 5; do
  [ -f "$LOG/c$c.done" ] && continue
  step "single-stream class $c, lr 0.01, 96k exposures, sampled eval"
  ( $PY cifar_main_stm_training.py --experiment-type few-shot --fine-classes $c --coarse-classes 0 1 \
      --batch-size 1 --training-steps 8000 --epochs 96 --evaluate-epochs 8 --evaluate-steps 1600 \
      --learning-rate 0.01 --ltm-checkpoint $CK/cifar_100_subclasses_12_e13.pth \
      --stm-checkpoint $CK/variants/stm_rl_fixed_e13_pt12.pth --run-root runs_stream_full \
      > "$LOG/c$c.log" 2>&1 && touch "$LOG/c$c.done" && step "class $c finished" || step "FAILED: class $c" ) &
done
wait
step "ALL DONE"
