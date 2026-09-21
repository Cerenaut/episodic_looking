#!/bin/bash
# Streaming (single-stream, batch size 1) runs of the two e13 STM variants, queued behind pt40 on the M3 (Gideon, 2026-09-20 15:50:
# M3 only, start after pt40). Protocol = the heads' streaming protocol: one unseen class, batch 1, 12 reported epochs of 1,000
# exposures (8,000 batch-1 steps at 8 steps per episode), evaluation of the four test sets after every epoch with 1,600 steps
# (200 episodes = one pass over the 200 test images at batch 1). Starts from the batch-16 pre-trained checkpoints used for the
# continual runs (the paper pre-trained at batch 1 for its streaming figure). Six runs in parallel (batch-1 jobs are latency-bound).
# Results: runs_stream_ref_fixed/ and runs_stream_a1_diff_lr0.01/ (few-shot_[c]_500/**/results_few-shot.txt); logs runs_local/streaming/.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/streaming
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
COMMON="--experiment-type few-shot --coarse-classes 0 1 --batch-size 1 --training-steps 8000 --epochs 12 --evaluate-steps 1600 --ltm-checkpoint $E13"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

step "waiting for pt40 (run_m3_pt40.sh) to finish"
while pgrep -f "^/bin/bash ./run_m3_pt40.sh" > /dev/null; do sleep 120; done
step "M3 free; starting streaming runs"
for c in 3 4 5; do
  [ -f "$LOG/ref_fixed_$c.done" ] || { step "streaming fixed RL class $c"; ( $PY cifar_main_stm_training.py $COMMON --fine-classes $c --eval-bias mean \
      --stm-checkpoint $CK/variants/stm_rl_fixed_e13_pt12.pth --run-root runs_stream_ref_fixed > "$LOG/ref_fixed_$c.log" 2>&1 \
      && touch "$LOG/ref_fixed_$c.done" && step "fixed RL class $c finished" || step "FAILED: fixed RL class $c" ) & }
  [ -f "$LOG/a1_$c.done" ] || { step "streaming A.1 lr 0.01 class $c"; ( $PY cifar_main_stm_training.py $COMMON --fine-classes $c --actor-training differentiable --learning-rate 0.01 \
      --stm-checkpoint $CK/variants/stm_a1_diff_lr0.01_e13_pt12.pth --run-root runs_stream_a1_diff_lr0.01 > "$LOG/a1_$c.log" 2>&1 \
      && touch "$LOG/a1_$c.done" && step "A.1 class $c finished" || step "FAILED: A.1 class $c" ) & }
done
wait
step "ALL DONE"
